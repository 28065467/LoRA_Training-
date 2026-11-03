"""Standalone training + eval pipeline — the .py version of notebooks/03's cells.

Same steps, same order, as notebook 03 (config -> fresh-start archive -> write recipe.yaml ->
launch training via the `automodel` CLI -> plot loss -> base-model eval cache -> final eval),
with one deliberate change: the **checkpoint-screening step is commented out** (see
STEP: CHECKPOINT SCREENING below) — this run always evaluates `LATEST` directly instead of
probing every saved checkpoint and picking a winner by similarity. Re-enable that block (it's
left in, just commented, not deleted) if you want screening back.

Every CONFIG value below can be overridden by an environment variable of the same name,
so you don't have to edit this file per run — see ../train.sh for a CLI wrapper that does
exactly that. Editing the file directly still works too (env vars just take priority).

    python -u scripts/run_training_pipeline.py \
        2>&1 | tee /path/to/OUTPUT_DIR/training_pipeline.log
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


# ============================== CONFIG — edit per run, or set env vars (see train.sh) ======
MODEL_ID = _env_str("MODEL_ID", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# No sensible generic default for these three — must be set via env var (train.sh requires
# them) or edited in directly below.
TRAIN_DATA_PATH = _env_str("TRAIN_DATA_PATH", "")
VAL_DATA_PATH = _env_str("VAL_DATA_PATH", "")
TEST_DATA_PATH = _env_str("TEST_DATA_PATH", "")

OUTPUT_DIR = _env_str("OUTPUT_DIR", "")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "recipe.yaml") if OUTPUT_DIR else ""
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints") if OUTPUT_DIR else ""
EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json") if OUTPUT_DIR else ""
BASE_EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "base_eval_results.json") if OUTPUT_DIR else ""

# True: archive whatever's currently in CHECKPOINT_DIR (if anything) to a timestamped backup
#   dir and start from scratch. Safe default whenever the recipe changed (dim/alpha/epochs/...).
# False: leave CHECKPOINT_DIR alone, so `automodel` auto-resumes from LATEST if present.
FRESH_START = _env_bool("FRESH_START", True)

# Screening is disabled this run (see module docstring) — always evaluate LATEST.
EVAL_CHECKPOINT = _env_str("EVAL_CHECKPOINT", "LATEST")

EVAL_MAX_NEW_TOKENS = _env_int("EVAL_MAX_NEW_TOKENS", 4096)
PROMPT_TOKEN_LIMIT = _env_int("PROMPT_TOKEN_LIMIT", 6144)     # should match SEQ_LENGTH below
FINAL_REPETITION_PENALTY = _env_float("FINAL_REPETITION_PENALTY", 1.1)

LORA_DIM = _env_int("LORA_DIM", 32)
LORA_ALPHA = _env_int("LORA_ALPHA", 64)
LORA_DROPOUT = _env_float("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES = ["*"]
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.05)
LEARNING_RATE = _env_float("LEARNING_RATE", 1e-4)
NUM_EPOCHS = _env_int("NUM_EPOCHS", 2)
# Fixed decay-step counts don't auto-scale with total steps — for a small train set (few
# hundred examples or fewer), raise NUM_EPOCHS and shrink WSD_DECAY_STEPS so the LR schedule
# isn't degenerate (aim for roughly 15-20% of total steps, e.g. 80/474 on a 3800-example run).
WSD_DECAY_STEPS = _env_int("WSD_DECAY_STEPS", 80)
SEQ_LENGTH = _env_int("SEQ_LENGTH", 6144)
LOCAL_BATCH_SIZE = _env_int("LOCAL_BATCH_SIZE", 4)
GLOBAL_BATCH_SIZE = _env_int("GLOBAL_BATCH_SIZE", 16)
NPROC_PER_NODE = _env_int("NPROC_PER_NODE", 1)

CUDA_VISIBLE_DEVICES = _env_str("CUDA_VISIBLE_DEVICES", "0")
CONDA_ENV = _env_str("CONDA_ENV", "nemo")
# ======================================================================================


def _check_required_config():
    missing = [name for name, val in
               [("TRAIN_DATA_PATH", TRAIN_DATA_PATH), ("VAL_DATA_PATH", VAL_DATA_PATH),
                ("TEST_DATA_PATH", TEST_DATA_PATH), ("OUTPUT_DIR", OUTPUT_DIR)]
               if not val]
    if missing:
        sys.exit(
            f"Missing required config: {', '.join(missing)}. "
            f"Set these as env vars (see ../train.sh) or edit the CONFIG block in this file."
        )


def run_isolated(*args, **kwargs):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        # PyTorch's own suggestion when a CUDA OOM shows a lot of memory "reserved but
        # unallocated" (allocator fragmentation) — notebook 06 hit this at dim=32/seq_length=6144
        # and fixed it with this flag; carrying the same fix over here.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    cmd = ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, **kwargs)


def run_script(script, *args, **kwargs):
    return run_isolated("python", "-u", script, *args, **kwargs)


def step_fresh_start_archive():
    if FRESH_START and os.path.isdir(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        backup_dir = f"{CHECKPOINT_DIR}_backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(CHECKPOINT_DIR, backup_dir)
        print(f"FRESH_START=True: archived existing checkpoints to {backup_dir}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def step_write_recipe():
    recipe = {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        "step_scheduler": {
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "ckpt_every_steps": 50,
            "val_every_steps": 10,
            "num_epochs": NUM_EPOCHS,
        },
        "dist_env": {"backend": "nccl", "timeout_minutes": 20},
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": 42,
            "ranked": True,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": MODEL_ID,
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "backend": {
                "_target_": "nemo_automodel.components.models.common.BackendConfig",
                "attn": "te",
                "linear": "torch",
                "rms_norm": "torch_fp32",
                "experts": "torch_mm",
                "dispatcher": "torch",
                "fake_balanced_gate": False,
                "enable_hf_state_dict_adapter": True,
                "enable_fsdp_optimizations": True,
            },
        },
        "peft": {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "target_modules": LORA_TARGET_MODULES,
            "dim": LORA_DIM,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": CHECKPOINT_DIR,
            "model_save_format": "safetensors",
            "save_consolidated": "final",
        },
        "distributed": {
            "strategy": "fsdp2",
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "sequence_parallel": False,
            "activation_checkpointing": True,
        },
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
            "path_or_dataset_id": TRAIN_DATA_PATH,
            "seq_length": SEQ_LENGTH,
            "truncation": True,
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "shuffle": True,
        },
        "validation_dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
            "path_or_dataset_id": VAL_DATA_PATH,
            "seq_length": SEQ_LENGTH,
            "truncation": True,
        },
        "validation_dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "_target_": "nemo_automodel.components.optim.optimizer.LRSchedulerConfig",
            "lr_decay_style": "WSD",
            "wsd_decay_steps": WSD_DECAY_STEPS,
            "lr_wsd_decay_style": "cosine",
            "min_lr": LEARNING_RATE * 0.1,
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(recipe, f, sort_keys=False)
    print(f"Wrote recipe to {CONFIG_PATH}")


def step_train():
    run_isolated("automodel", CONFIG_PATH, "--nproc-per-node", str(NPROC_PER_NODE), check=True)


def step_plot_loss():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — this runs with no display attached
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available in this interpreter — skipping loss plot.")
        return

    def _read_jsonl(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    train_records = _read_jsonl(f"{CHECKPOINT_DIR}/training.jsonl")
    val_records = _read_jsonl(f"{CHECKPOINT_DIR}/validation.jsonl")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([r["step"] for r in train_records], [r["loss"] for r in train_records],
            color="#2a78d6", linewidth=2, marker="o", markersize=4, zorder=3, label="train loss")
    ax.plot([r["step"] for r in val_records], [r["val_loss"] for r in val_records],
            color="#eb6834", linewidth=2, marker="o", markersize=4, zorder=3, label="val loss")
    ax.set_title("Training / Validation Loss", fontsize=13, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, axis="y", color="#e3e2dd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name not in ("top", "right"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)

    out_path = f"{CHECKPOINT_DIR}/training_loss.png"
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved loss plot to {out_path}")


def step_base_eval_cache():
    if os.path.exists(BASE_EVAL_RESULTS_PATH):
        print(f"{BASE_EVAL_RESULTS_PATH} already exists — reusing it.")
        return
    run_script(
        os.path.join(SCRIPTS_DIR, "base_eval.py"),
        "--model-id", MODEL_ID,
        "--eval-data", TEST_DATA_PATH,
        "--out", BASE_EVAL_RESULTS_PATH,
        "--prompt-token-limit", str(PROMPT_TOKEN_LIMIT),
        "--max-new-tokens", str(EVAL_MAX_NEW_TOKENS),
        check=True,
    )


# ============================================================================
# STEP: CHECKPOINT SCREENING — disabled for this run.
#
# Normally this probes every saved checkpoint against one representative TEST_DATA_PATH
# example and picks the winner by char-similarity (see notebooks/scripts/run_post_training_
# pipeline.py for a standalone version of this exact logic, including the caveat about ties
# when the probe example is too easy to discriminate between checkpoints). Commented out here
# per request — EVAL_CHECKPOINT is hardcoded to "LATEST" above instead. Uncomment and call
# this if you want screening back.
# ============================================================================
# def step_checkpoint_screening():
#     import difflib
#     import re
#
#     PROBE_TEST_INDEX = 0
#     PROBE_MAX_NEW_TOKENS = 1024
#     PROBE_CLEANUP_MERGED = True
#
#     step_dirs = sorted(
#         (d for d in os.listdir(CHECKPOINT_DIR)
#          if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and re.match(r"epoch_\d+_step_\d+$", d)),
#         key=lambda d: int(re.search(r"step_(\d+)$", d).group(1)),
#     )
#     probe_results = {}
#     for step_dir in step_dirs:
#         probe_path = os.path.join(CHECKPOINT_DIR, step_dir, f"probe_index{PROBE_TEST_INDEX}.json")
#         if not os.path.exists(probe_path):
#             run_script(
#                 os.path.join(SCRIPTS_DIR, "reload_and_eval.py"),
#                 "--model-id", MODEL_ID,
#                 "--checkpoint-dir", CHECKPOINT_DIR,
#                 "--checkpoint-name", step_dir,
#                 "--out", probe_path,
#                 "--eval-data", TEST_DATA_PATH,
#                 "--eval-indices", str(PROBE_TEST_INDEX),
#                 "--prompt-token-limit", str(PROMPT_TOKEN_LIMIT),
#                 "--max-new-tokens", str(PROBE_MAX_NEW_TOKENS),
#                 "--base-cache", BASE_EVAL_RESULTS_PATH,
#                 "--loader", "nemo",
#                 "--skip-toy-prompt",
#                 check=True,
#             )
#         with open(probe_path, encoding="utf-8") as f:
#             r = json.load(f)["eval"]["results"][0]
#         score = difflib.SequenceMatcher(None, r["adapter_text"], r["reference"] or "").ratio()
#         probe_results[step_dir] = score
#
#     winner = max(probe_results, key=probe_results.get)
#     if PROBE_CLEANUP_MERGED:
#         import shutil as _shutil
#         for step_dir in step_dirs:
#             if step_dir == winner:
#                 continue
#             merged_dir = os.path.join(CHECKPOINT_DIR, step_dir, "merged_hf")
#             if os.path.isdir(merged_dir):
#                 _shutil.rmtree(merged_dir)
#     return winner


def step_final_eval():
    run_script(
        os.path.join(SCRIPTS_DIR, "reload_and_eval.py"),
        "--model-id", MODEL_ID,
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--checkpoint-name", EVAL_CHECKPOINT,
        "--out", EVAL_RESULTS_PATH,
        "--eval-data", TEST_DATA_PATH,
        "--prompt-token-limit", str(PROMPT_TOKEN_LIMIT),
        "--max-new-tokens", str(EVAL_MAX_NEW_TOKENS),
        "--base-cache", BASE_EVAL_RESULTS_PATH,
        "--loader", "nemo",
        "--repetition-penalty", str(FINAL_REPETITION_PENALTY),
        check=True,
    )


def step_print_results():
    import difflib

    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        eval_results = json.load(f)

    print("\nCheckpoint:", eval_results["consolidated_dir"])
    print("Loader:", eval_results.get("loader", "peft"), "| has adapter:", eval_results["has_adapter"])

    ev = eval_results.get("eval")
    if not ev:
        return
    if ev["skipped"]:
        print(f"Skipped {len(ev['skipped'])} prompt(s) over {ev['prompt_token_limit']} tokens")
    print(f"{ev['n_differ']}/{ev['n_total']} eval prompts: adapter output differs from base output")

    scores = [
        difflib.SequenceMatcher(None, r["adapter_text"], r["reference"] or "").ratio()
        for r in ev["results"]
    ]
    if scores:
        print(f"Mean char-similarity ({EVAL_CHECKPOINT}): {sum(scores) / len(scores):.4f}")


def main():
    _check_required_config()
    step_fresh_start_archive()
    step_write_recipe()
    step_train()
    step_plot_loss()
    step_base_eval_cache()
    # winner = step_checkpoint_screening()   # disabled — see block above
    step_final_eval()
    step_print_results()
    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main())
