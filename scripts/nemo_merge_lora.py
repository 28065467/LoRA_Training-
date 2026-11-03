"""Load a NeMo AutoModel LoRA checkpoint with NeMo's own loader, merge the adapter into the base
weights, and export a plain Hugging Face checkpoint for fast (KV-cached) HF generation.

Why this exists: NeMo saves MoE expert LoRA (GroupedExpertsLoRA) in its own packed layout, which
`peft.PeftModel.from_pretrained` silently ignores — only attention/lm_head LoRA gets applied. This
script instead rebuilds the model exactly the way the training recipe's auto-resume does (same
config.yaml, same Checkpointer.load_model path), so every trained LoRA weight is loaded, then folds
each adapter into its base weight:
    LinearLoRA:         W       += scale * B @ A                     (nn.Linear layout, [out, in])
    GroupedExpertsLoRA: W[e]    += scale * A[e] @ B[e]               (grouped layout, x @ W[e])
NeMo's custom model has no KV cache, so generating from it directly would be O(n^2) per token;
the merged HF export generates at normal speed and is cached next to the checkpoint for reuse.

Single process, single GPU (pin with CUDA_VISIBLE_DEVICES). Usage:
    python nemo_merge_lora.py --checkpoint ../outputs/.../checkpoints/LOWEST_VAL --out <dir>
"""
import argparse
import glob
import json
import os
import shutil

# The recipe calls torch.distributed.init_process_group(env://) — provide a 1-rank world when
# not launched through torchrun.
for _k, _v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29533"}.items():
    os.environ.setdefault(_k, _v)

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import _maybe_adapt_state_dict_to_hf
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

SHARD_BYTES = 5 * 1024**3


def _local(t):
    # FSDP2 on a 1-rank mesh: the local shard *is* the full tensor.
    return t.to_local() if isinstance(t, DTensor) else t


@torch.no_grad()
def merge_lora_in_place(model) -> tuple[int, int]:
    n_linear = n_experts = 0
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_gate_and_up_A"):
            # B is zero-initialized, so an all-zero B means the checkpoint's expert LoRA never
            # made it into the model (merge would silently be a no-op).
            for b in ("lora_gate_and_up_B", "lora_down_B"):
                if not _local(getattr(mod, b)).any():
                    raise RuntimeError(f"{name}.{b} is all zeros — expert LoRA was not loaded")
            scale = mod.scale
            for base, a, b in (("gate_and_up_projs", "lora_gate_and_up_A", "lora_gate_and_up_B"),
                               ("down_projs", "lora_down_A", "lora_down_B")):
                w = _local(getattr(mod, base))
                delta = torch.bmm(_local(getattr(mod, a)).float(), _local(getattr(mod, b)).float())
                w.copy_((w.float() + scale * delta).to(w.dtype))
            n_experts += 1
        elif isinstance(getattr(mod, "lora_A", None), torch.nn.Module) and hasattr(mod, "weight"):
            if getattr(mod, "use_dora", False):
                raise NotImplementedError(f"{name}: DoRA merge not supported")
            w = _local(mod.weight)
            delta = _local(mod.lora_B.weight).float() @ _local(mod.lora_A.weight).float()
            w.copy_((w.float() + mod.scale * delta).to(w.dtype))
            n_linear += 1
    return n_linear, n_experts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint step dir (e.g. .../checkpoints/LOWEST_VAL) containing config.yaml + model/")
    parser.add_argument("--out", required=True, help="Directory to write the merged HF checkpoint to")
    args = parser.parse_args()

    ckpt_dir = os.path.realpath(args.checkpoint)
    cfg = load_yaml_config(os.path.join(ckpt_dir, "config.yaml"))
    cfg.set_by_dotted("checkpoint.checkpoint_dir", os.path.dirname(ckpt_dir))
    cfg.set_by_dotted("checkpoint.restore_from", ckpt_dir)
    model_id = cfg.get("model.pretrained_model_name_or_path")

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()  # builds base model + LoRA, then Checkpointer.load_model(<ckpt>/model)
    assert torch.distributed.get_world_size() == 1, "merge assumes a single-rank (unsharded) model"
    model = recipe.model_parts[0]

    n_linear, n_experts = merge_lora_in_place(model)
    print(f"Merged LoRA into {n_linear} linear module(s) and {n_experts} grouped-expert module(s)")
    if n_linear == 0 and n_experts == 0:
        raise RuntimeError("No LoRA modules found on the loaded model")

    base_dir = snapshot_download(model_id, allow_patterns=["*.json", "*.jinja", "*.txt", "*.py"])
    with open(os.path.join(base_dir, "model.safetensors.index.json")) as f:
        base_keys = set(json.load(f)["weight_map"])

    # The state-dict adapter re-detects the HF "model." prefix on every from_hf() call; loading the
    # PEFT adapter (keys start with "base_model.model.") flips it off, which would drop the prefix
    # from exported expert keys. Reset it from the base checkpoint's own key layout.
    model.state_dict_adapter._uses_model_prefix = any(k.startswith("model.") for k in base_keys)

    state_dict = {k: _local(v) for k, v in model.state_dict().items() if "lora_" not in k}
    hf_sd = _maybe_adapt_state_dict_to_hf(model, state_dict, device_mesh=recipe.moe_mesh)
    hf_sd = {k: v.detach().to("cpu").contiguous() for k, v in hf_sd.items()}
    del state_dict, model, recipe
    torch.cuda.empty_cache()

    # Exported keys must be exactly the base checkpoint's keys, or HF loading would silently
    # leave weights at random init.
    missing, extra = base_keys - set(hf_sd), set(hf_sd) - base_keys
    if missing or extra:
        raise RuntimeError(f"Key mismatch vs base checkpoint: missing={sorted(missing)[:5]} "
                           f"(+{len(missing)}), extra={sorted(extra)[:5]} (+{len(extra)})")

    tmp_out = args.out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp_out, ignore_errors=True)
    os.makedirs(tmp_out)
    shards, cur, cur_bytes = [], {}, 0
    for k in sorted(hf_sd):
        nbytes = hf_sd[k].numel() * hf_sd[k].element_size()
        if cur and cur_bytes + nbytes > SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = hf_sd[k]
        cur_bytes += nbytes
    shards.append(cur)

    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(tmp_out, fname), metadata={"format": "pt"})
        for k, v in shard.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
        print(f"Wrote {fname}")
    with open(os.path.join(tmp_out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    # config / generation config from the base model; tokenizer + chat template from the checkpoint
    for fname in ("config.json", "generation_config.json"):
        shutil.copy(os.path.join(base_dir, fname), tmp_out)
    for src in glob.glob(os.path.join(ckpt_dir, "model", "*")):
        name = os.path.basename(src)
        if name.startswith("adapter_") or name.startswith("automodel_peft"):
            continue
        shutil.copy(src, tmp_out)

    shutil.rmtree(args.out, ignore_errors=True)
    os.rename(tmp_out, args.out)
    print(f"Merged HF checkpoint written to {args.out}")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
