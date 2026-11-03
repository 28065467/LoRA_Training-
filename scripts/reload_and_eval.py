"""Reload a trained checkpoint and run inference, entirely in a standalone process.

Run via subprocess from a notebook so the notebook's own ipykernel process never imports
torch / touches the GPU — all VRAM this script allocates is released the instant it exits,
instead of being held by the kernel for the rest of the notebook session. Results are
written to a JSON file for the calling notebook to read back and display (no torch needed
for that part).

Usage:
    python reload_and_eval.py \
        --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --checkpoint-dir ../outputs/.../checkpoints \
        [--checkpoint-name LATEST]   # or LOWEST_VAL, or an epoch_N_step_M dir name
        --out ../outputs/.../eval_results.json \
        [--eval-data ../all_chatml_testing.json] \
        [--prompt-token-limit 2048] \
        [--max-new-tokens 256] \
        [--toy-prompt "Write a Python function that reverses a string."] \
        [--skip-toy-prompt] \
        [--eval-indices 4,9]          # only these eval-data indices (default: all)
        [--base-cache ../outputs/.../base_eval_results.json] \
        [--loader peft]              # or nemo — see below
        [--repetition-penalty 1.0]   # >1.0 discourages repeating already-generated tokens

--base-cache reuses base_eval.py's output instead of regenerating the base (no-adapter) text —
the base model doesn't change between checkpoints, so re-running its generation on every single
eval call is pure waste; this roughly halves eval time once the cache exists (skips the
disable_adapter() generation entirely for any prompt found in the cache by index).

--loader nemo is for NeMo AutoModel checkpoints: NeMo saves MoE expert LoRA in its own packed
layout that PeftModel silently drops (only attention/lm_head LoRA would apply). Instead,
scripts/nemo_merge_lora.py loads the checkpoint with NeMo's own loader, merges every adapter into
the base weights and exports a plain HF checkpoint to <checkpoint>/merged_hf (reused on later
runs), which is then loaded here. There's no separate base model in that mode, so base_text only
comes from --base-cache (uncached prompts, incl. the toy prompt, get base_text=None).
"""
import argparse
import json
import os
import subprocess
import sys

import torch

from model_loading import load_causal_model, load_tokenizer


def find_checkpoint_dir(checkpoint_dir: str, checkpoint_name: str = "LATEST") -> str:
    ckpt_link = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(ckpt_link):
        raise FileNotFoundError(
            f"No '{checkpoint_name}' entry under {checkpoint_dir} — run "
            f"os.listdir({checkpoint_dir!r}) manually and check the path."
        )
    return os.path.join(ckpt_link, "model")


def ensure_nemo_merged(checkpoint_dir: str, checkpoint_name: str) -> str:
    step_dir = os.path.realpath(os.path.join(checkpoint_dir, checkpoint_name))
    merged_dir = os.path.join(step_dir, "merged_hf")
    if os.path.exists(os.path.join(merged_dir, "model.safetensors.index.json")):
        print(f"Reusing merged checkpoint: {merged_dir}")
    else:
        # Separate process so the NeMo model's VRAM is fully released before HF loads the merge.
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemo_merge_lora.py")
        subprocess.run([sys.executable, "-u", script, "--checkpoint", step_dir, "--out", merged_dir], check=True)
    return merged_dir


def load_model_and_tokenizer(model_id: str, consolidated_dir: str):
    tokenizer = load_tokenizer(model_id)
    has_adapter = os.path.exists(os.path.join(consolidated_dir, "adapter_config.json"))

    if has_adapter:
        from peft import PeftModel

        base = load_causal_model(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, consolidated_dir)
    else:
        model = load_causal_model(
            consolidated_dir, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
        )
    return model, tokenizer, has_adapter


def generate(model, tokenizer, prompt_msgs, max_new_tokens, has_adapter, cached_base_text=None,
             repetition_penalty=1.0):
    # has_adapter: base_text can be generated via disable_adapter() when not cached.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(model.device)

    with torch.no_grad():
        adapter_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=repetition_penalty)
    adapter_text = tokenizer.decode(adapter_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)

    base_text = cached_base_text
    if has_adapter and base_text is None:
        with torch.no_grad():
            with model.disable_adapter():
                base_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                           repetition_penalty=repetition_penalty)
        base_text = tokenizer.decode(base_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)

    return adapter_text, base_text, prompt_ids.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-name", default="LATEST",
                        help="Entry under --checkpoint-dir to load: LATEST, LOWEST_VAL, or an epoch_N_step_M dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-data", default=None, help="Path to a JSON array of {messages: [...]} records")
    parser.add_argument("--prompt-token-limit", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--toy-prompt", default="Write a Python function that reverses a string.")
    parser.add_argument("--skip-toy-prompt", action="store_true",
                        help="Skip the toy-prompt generation — useful when screening many checkpoints "
                             "against a small --eval-indices subset, where it'd be pure overhead")
    parser.add_argument("--eval-indices", default=None,
                        help="Comma-separated indices into --eval-data to actually run (e.g. '4,9'); "
                             "default runs every record. For cheaply probing one representative prompt "
                             "across many checkpoints instead of the full eval set")
    parser.add_argument("--loader", choices=["peft", "nemo"], default="peft",
                        help="peft: base + PeftModel adapter. nemo: NeMo-loaded, merged checkpoint "
                             "(required for NeMo MoE expert LoRA)")
    parser.add_argument("--base-cache", default=None,
                        help="Path to base_eval.py's output JSON; reuses its base_text (matched by "
                             "eval-set index) instead of regenerating it here")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="HF generate()'s repetition_penalty; 1.0 = off (default, unchanged "
                             "behavior). >1.0 discourages repeating already-generated tokens — a "
                             "candidate fix for greedy decoding looping until max_new_tokens instead "
                             "of emitting EOS. Try ~1.15.")
    args = parser.parse_args()

    base_cache = {}
    if args.base_cache and os.path.exists(args.base_cache):
        with open(args.base_cache, encoding="utf-8") as f:
            base_cache = {r["index"]: r["base_text"] for r in json.load(f)["results"]}
        print(f"Loaded {len(base_cache)} cached base output(s) from {args.base_cache}")

    if args.loader == "nemo":
        consolidated_dir = ensure_nemo_merged(args.checkpoint_dir, args.checkpoint_name)
    else:
        consolidated_dir = find_checkpoint_dir(args.checkpoint_dir, args.checkpoint_name)
    print(f"Loading checkpoint: {consolidated_dir}")
    model, tokenizer, has_adapter = load_model_and_tokenizer(args.model_id, consolidated_dir)
    print(f"Loaded ({'base + LoRA adapter' if has_adapter else 'merged/standalone checkpoint'}).")
    # A merged NeMo checkpoint is still a fine-tuned model — compare against cached base outputs.
    compare_base = has_adapter or args.loader == "nemo"

    def differs(adapter_text, base_text):
        return (adapter_text != base_text) if compare_base and base_text is not None else None

    output = {"consolidated_dir": consolidated_dir, "loader": args.loader, "has_adapter": has_adapter}

    if args.skip_toy_prompt:
        output["toy_prompt"] = None
    else:
        # Toy prompt — quick "is the adapter actually contributing" sanity check.
        toy_adapter, toy_base, _ = generate(
            model, tokenizer, [{"role": "user", "content": args.toy_prompt}], args.max_new_tokens, has_adapter,
            repetition_penalty=args.repetition_penalty,
        )
        output["toy_prompt"] = {
            "prompt": args.toy_prompt,
            "adapter_text": toy_adapter,
            "base_text": toy_base,
            "differs": differs(toy_adapter, toy_base),
        }
        toy_differs = output["toy_prompt"]["differs"]
        print(f"Toy-prompt adapter output == base output: {'n/a' if toy_differs is None else not toy_differs}")

    # Real eval set, if given.
    if args.eval_data:
        with open(args.eval_data, encoding="utf-8") as f:
            records = json.load(f)

        wanted_indices = None
        if args.eval_indices is not None:
            wanted_indices = {int(x) for x in args.eval_indices.split(",") if x.strip() != ""}
        total_considered = len(records) if wanted_indices is None else len(wanted_indices)

        results, skipped = [], []
        for i, rec in enumerate(records):
            if wanted_indices is not None and i not in wanted_indices:
                continue
            msgs = rec["messages"]
            prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
            reference = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else None

            probe_ids = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(probe_ids) > args.prompt_token_limit:
                skipped.append({"index": i, "num_tokens": len(probe_ids)})
                continue

            adapter_text, base_text, _ = generate(
                model, tokenizer, prompt_msgs, args.max_new_tokens, has_adapter,
                cached_base_text=base_cache.get(i),
                repetition_penalty=args.repetition_penalty,
            )
            results.append({
                "index": i,
                "uuid": rec.get("uuid"),
                "differs": differs(adapter_text, base_text),
                "adapter_text": adapter_text,
                "base_text": base_text,
                "reference": reference,
            })
            print(f"[{i}] done ({len(results)}/{total_considered - len(skipped)})")

        n_differ = sum(1 for r in results if r["differs"])
        output["eval"] = {
            "eval_data": args.eval_data,
            "prompt_token_limit": args.prompt_token_limit,
            "skipped": skipped,
            "n_total": len(results),
            "n_differ": n_differ,
            "results": results,
        }
        print(f"Skipped {len(skipped)} prompt(s) over {args.prompt_token_limit} tokens: {skipped}")
        print(f"{n_differ}/{len(results)} eval prompts: adapter output differs from base output")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
