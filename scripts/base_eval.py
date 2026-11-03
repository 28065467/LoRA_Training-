"""Run the ORIGINAL, untrained base model over an eval set — no adapter, no checkpoint.

This is the "Freeze Baseline" / "Base Evaluation" step (PDF section C): before comparing
against a fine-tuned adapter, record what the untouched base model produces on the same
held-out prompts. Run via subprocess (same pattern as scripts/reload_and_eval.py) so the
calling notebook's ipykernel process never imports torch — VRAM is released the instant this
process exits.

Usage:
    python base_eval.py \
        --model-id Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --eval-data ../all_chatml_testing.json \
        --out ../outputs/.../base_eval_results.json \
        [--prompt-token-limit 4096] \
        [--max-new-tokens 256] \
        [--repetition-penalty 1.0]   # >1.0 discourages repeating already-generated tokens
"""
import argparse
import json
import os

import torch

from model_loading import load_causal_model, load_tokenizer


def generate(model, tokenizer, prompt_msgs, max_new_tokens, repetition_penalty=1.0):
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=repetition_penalty)
    return tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True), prompt_ids.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--eval-data", required=True, help="Path to a JSON array of {messages: [...]} records")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-token-limit", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="HF generate()'s repetition_penalty; 1.0 = off (default, unchanged behavior)")
    args = parser.parse_args()

    print(f"Loading base model (no adapter): {args.model_id}")
    tokenizer = load_tokenizer(args.model_id)
    model = load_causal_model(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    print("Loaded.")

    with open(args.eval_data, encoding="utf-8") as f:
        records = json.load(f)

    results, skipped = [], []
    for i, rec in enumerate(records):
        msgs = rec["messages"]
        prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
        reference = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else None

        probe_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(probe_ids) > args.prompt_token_limit:
            skipped.append({"index": i, "num_tokens": len(probe_ids)})
            continue

        base_text, _ = generate(model, tokenizer, prompt_msgs, args.max_new_tokens,
                                 repetition_penalty=args.repetition_penalty)
        results.append({
            "index": i,
            "uuid": rec.get("uuid"),
            "base_text": base_text,
            "reference": reference,
        })
        print(f"[{i}] done ({len(results)}/{len(records) - len(skipped)})")

    output = {
        "model_id": args.model_id,
        "eval_data": args.eval_data,
        "prompt_token_limit": args.prompt_token_limit,
        "skipped": skipped,
        "n_total": len(results),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if skipped:
        print(f"Skipped {len(skipped)} prompt(s) over {args.prompt_token_limit} tokens: {skipped}")
    print(f"Ran base eval on {len(results)} prompts. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
