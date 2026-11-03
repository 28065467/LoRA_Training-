# Qwen3-Coder LoRA 訓練/合併/Eval 核心流程

這份資料夾只放「一定會用到」的5個腳本——訓練、合併LoRA、評估。其他像CodeBERTScore/ICE-Score/CodeJudge這類額外的品質評估工具不在這裡，那些是研究用的輔助工具，不是訓練流程本身需要的。

## 目錄結構

```
core_pipeline/
├── README.md              本檔
├── requirements.txt        已驗證可用的套件版本
├── train.sh                訓練的CLI wrapper，不用編輯Python檔案填參數
└── scripts/
    ├── run_training_pipeline.py   主流程：寫recipe.yaml → 呼叫nemo-automodel訓練 → 畫loss圖 → base model eval → 訓練後eval
    ├── nemo_merge_lora.py         把NeMo訓練出的LoRA adapter合併回base權重，輸出一份標準HF checkpoint
    ├── reload_and_eval.py        載入checkpoint（trained或merged）跑推理，輸出跟base/對照組的比較
    ├── base_eval.py               跑「完全沒訓練過」的base model當baseline，結果會被reload_and_eval.py快取重用
    └── model_loading.py           共用小工具：自動判斷該用AutoModelForCausalLM還是AutoModelForImageTextToText載入（VL架構的模型，如Qwen3.5系列，HF註冊在ImageTextToText底下，直接用CausalLM會載入失敗）
```

## 環境建置

```bash
conda create -n nemo python=3.11 -y
conda activate nemo

# torch要裝CUDA 13.0對應的build，不要用預設的pip install torch
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

**硬體需求**：這份pipeline是在2張H200（各143GB）上驗證過的；單張卡跑dim=32的LoRA、seq_length=6144沒問題，更大的seq_length或更大的base model（例如dense的Qwen3.5系列，不是MoE）記憶體壓力會明顯上升，必要時調小`LOCAL_BATCH_SIZE`。

**裝完先驗證GPU看得到**：
```bash
python scripts/check_cuda.py
```

## 怎麼跑

### 1. 訓練

**用`train.sh`，不用自己編輯Python檔案填參數**：

```bash
./train.sh \
  --train data/chatml_train.json \
  --val data/chatml_val.json \
  --test data/chatml_test.json \
  --output-dir outputs/my-run \
  --gpu 0 \
  --tmux my-run        # 背景跑在一個叫my-run的tmux session裡，不加這個就是前景跑
```

必填：`--train`/`--val`/`--test`（chatml格式JSON，每筆是`{"messages": [{"role": "system"/"user"/"assistant", "content": "..."}]}`）、`--output-dir`（**每個新設定都要給新目錄，不要共用**）。其他超參數（`--model-id`/`--dim`/`--alpha`/`--dropout`/`--epochs`/`--seq-length`/`--wsd-decay-steps`等）都有合理預設，要改再加對應的flag，完整列表看`./train.sh --help`。

`--gpu`：這台機器常常另一張卡被別人佔用，開跑前先`nvidia-smi`確認。

**資料量很小時要注意**：`WSD_DECAY_STEPS`不會自動跟著總步數調整，資料筆數少（例如<300筆）時，照預設可能根本訓不完一個完整的LR schedule，要用`--epochs`往上調、`--wsd-decay-steps`往下調（抓總步數的15~20%左右，總步數≈epochs×訓練筆數/global_batch）。

`train.sh`背後其實是幫你把這些值設成環境變數，再呼叫`scripts/run_training_pipeline.py`——如果不想用`train.sh`，直接設環境變數再跑該script也行，或照舊打開檔案編輯CONFIG區塊（env var設定會覆蓋檔案內的預設值）。

這一步內部會自動依序做：寫`recipe.yaml` → 呼叫`automodel`訓練 → 畫loss曲線 → 跑base model baseline（呼叫`base_eval.py`） → 跑訓練後checkpoint的eval（呼叫`reload_and_eval.py`，內部又會呼叫`nemo_merge_lora.py`）。不需要手動分開跑這些步驟。

### 2. 合併LoRA（通常不用手動跑）

`reload_and_eval.py --loader nemo`會自動呼叫這個，把NeMo格式的LoRA checkpoint合併成一份標準HF權重，存在`<checkpoint目錄>/merged_hf/`，之後重複eval會直接重用，不會每次重新合併。

**為什麼一定要用這個，不能直接用`peft.PeftModel.from_pretrained`**：NeMo訓練MoE模型時，expert的LoRA是用自己的packed格式存的，`PeftModel`載入時會「安靜地」忽略這部分、只套用到attention/lm_head的LoRA——不會報錯，但訓練效果會大部分消失而你不會發現。dense（非MoE）模型沒有這個問題，但統一都走這條路比較安全。

手動單獨跑的話：
```bash
python scripts/nemo_merge_lora.py --checkpoint <checkpoints目錄>/LATEST --out <輸出merged_hf的目錄>
```

### 3. Eval（通常也是`run_training_pipeline.py`自動呼叫，這裡列出手動用法）

```bash
# base model baseline（只需要跑一次，結果會被下面的eval快取重用）
python scripts/base_eval.py \
    --model-id <MODEL_ID> \
    --eval-data <test.json> \
    --out <OUTPUT_DIR>/base_eval_results.json

# 訓練後checkpoint
python scripts/reload_and_eval.py \
    --model-id <MODEL_ID> \
    --checkpoint-dir <OUTPUT_DIR>/checkpoints \
    --checkpoint-name LATEST \
    --eval-data <test.json> \
    --out <OUTPUT_DIR>/eval_results.json \
    --base-cache <OUTPUT_DIR>/base_eval_results.json \
    --loader nemo
```

`--checkpoint-name`可以是`LATEST`、`LOWEST_VAL`，或具體的`epoch_N_step_M`目錄名。

## 幾個踩過的坑

- **tmux可能不在PATH上**：如果`tmux: command not found`，用完整路徑（`/home/acd_rd7_admin/miniconda3/bin/tmux`）或在背景指令前`export PATH=.../miniconda3/condabin:.../miniconda3/bin:$PATH`。
- **OOM但記憶體看起來還有剩**：如果錯誤訊息顯示「reserved but unallocated」很多，是allocator碎片化，不是真的沒記憶體——在環境變數加`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（`run_training_pipeline.py`內部已經自動加了這個，但手動跑其他script時要自己注意）。
- **這台機器GPU是共用的**：另一張卡經常被別人的程序佔住，開跑前一定先`nvidia-smi`確認，不要假設兩張卡都是空的。
- **換了非標準的CausalLM架構（例如VL模型）要小心**：`model_loading.py`已經處理了「HF architecture只登記在`AutoModelForImageTextToText`底下」的狀況，但如果遇到更新的模型架構，先用`AutoConfig.from_pretrained(...).architectures`確認一下登記在哪個Auto class底下，再決定要不要載入失敗。
