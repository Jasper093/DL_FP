# MESC / SMES 4-Task Reasoning Codebase

This repository implements a simplified reproduction pipeline for **Sequential Multimodal Emotional Support (SMES)** reasoning on the **MESC** dataset.

The core idea is to model emotional-support dialogue as a structured 4-task chain:

```text
User turn + dialogue history + video/audio textual cues
        ↓
Task 1: User emotion recognition
        ↓
Task 2: Therapist strategy prediction
        ↓
Task 3: Therapist/system emotion prediction
        ↓
Task 4: Therapist response generation
```

Instead of directly generating a therapist response from dialogue context, the code trains/evaluates separate reasoning stages and allows chained LoRA adapters, ablations, and post-validation analysis.

---

## 1. What This Code Can Do

### Main capabilities

| Capability | Description |
|---|---|
| Preprocess MESC-style raw dialogue data | Converts raw dialogue JSON into task-wise JSONL files for the 4 reasoning tasks. |
| Train Task 1-3 classification models | Fine-tunes a causal LM with LoRA to predict user emotion, therapist strategy, or therapist emotion. |
| Train Task 4 response generation model | Fine-tunes a causal LM with LoRA to generate therapist responses. |
| Chain adapters across tasks | Later tasks can initialize from previous task adapters, e.g. Task 2 starts from Task 1 adapter. |
| Run classification evaluation/inference | Computes accuracy, weighted F1, macro F1, confusion matrix, prediction distribution, and ranking Excel files. |
| Run generation evaluation/inference | Generates responses and computes BLEU-2, ROUGE-L, optional BERTScore, teacher-forced NLL/PPL, and ranking Excel files. |
| Run ablation-style inference | Uses config flags to drop video cues, history, emotion fields, or strategy fields during evaluation/inference. |
| Export analysis artifacts | Saves `metrics.json`, `predictions.jsonl`, and Excel ranking workbooks for error analysis. |

---

## 2. Overall Workflow

```text
Given dataset
        |
        ▼
Raw MESC-style JSON
        │
        ▼
preprocess/preprocess_all.py
        │
        ├── data/processed/user_emotion/*.jsonl
        ├── data/processed/therapist_strategy/*.jsonl
        ├── data/processed/therapist_emotion/*.jsonl
        └── data/processed/therapist_response/*.jsonl
        │
        ▼
Training
        │
        ├── training/train_classification.py  → Task 1 / Task 2 / Task 3
        └── training/train_generation.py      → Task 4
        │
        ▼
runs/<task>/run_xxx/best_adapter
        │
        ▼
Evaluation / Inference
        │
        └── evaluation/run_eval.py
                │
                ├── metrics.json
                ├── predictions.jsonl
                ├── classification_rankings.xlsx
                └── generation_rankings.xlsx
```

---

## 3. Task Definitions

| Task | Config | Dataset folder | Model behavior | Output |
|---|---|---|---|---|
| Task 1 | `configs/task1.yaml` | `data/processed/user_emotion/` | Classification by generation | User emotion label |
| Task 2 | `configs/task2.yaml` | `data/processed/therapist_strategy/` | Classification by generation | Therapist strategy label |
| Task 3 | `configs/task3.yaml` | `data/processed/therapist_emotion/` | Classification by generation | Therapist/system emotion label |
| Task 4 | `configs/task4.yaml` | `data/processed/therapist_response/` | Text generation | Therapist response |

---

## 4. Directory Structure

```text
reasoning_tasks/
├── common/
│   ├── config_utils.py
│   └── dataset.py
│
├── concatenate/
│   └── func.py
│
├── configs/
│   ├── model.yaml
│   ├── preprocess.yaml
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   └── task4.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── inf_audio_raw/
│
├── evaluation/
│   ├── run_eval.py
│   ├── loaders.py
│   ├── export_utils.py
│   ├── record_utils.py
│   ├── excel_rankings.py
│   ├── classification/
│   └── generation/
│
├── preprocess/
│   ├── preprocess_all.py
│   ├── core.py
│   └── utils.py
│
├── prompts/
│   ├── user_emotion_prompt.txt
│   ├── therapist_strategy_prompt.txt
│   ├── therapist_emotion_prompt.txt
│   ├── therapist_response_v_prompt.txt
│   └── therapist_response_vp_prompt.txt
│
├── training/
│   ├── train_classification.py
│   ├── train_generation.py
│   └── trainer_utils.py
│
└── runs/
    └── <task_name>/run_xxx/
```

---

## 5. How to Run

Run commands from the `reasoning_tasks/` directory unless stated otherwise.

### 5.0 raw data concatenation
```bash
python -m concatenate.func
```

### 5.1 Preprocess data

```bash
python -m preprocess.preprocess_all.py --config configs/preprocess.yaml
```

Uncomment the train/test lines if full preprocessing is needed.

---

### 5.2 Train classification tasks

Task 1: user emotion

```bash
python -m training.train_classification \
  --config configs/task1.yaml \
  --mdl_config configs/model.yaml
```

Task 2: therapist strategy

```bash
python -m training.train_classification \
  --config configs/task2.yaml \
  --mdl_config configs/model.yaml
```

Task 3: therapist emotion

```bash
python -m training.train_classification \
  --config configs/task3.yaml \
  --mdl_config configs/model.yaml
```

The best LoRA adapter is saved to:

```text
runs/<task_name>/run_001/best_adapter/
```

---

### 5.3 Train response generation task

Task 4: therapist response generation

```bash
python -m training.train_generation \
  --config configs/task4.yaml \
  --mdl_config configs/model.yaml
```

The best adapter is selected by lowest validation loss.

---

### 5.4 Run evaluation / inference

Unified launcher:

```bash
python -m evaluation.run_eval \
  --config configs/task1.yaml \
  --mdl_config configs/model.yaml
```

Change the task config to evaluate another task:

```bash
python -m evaluation.run_eval --config configs/task2.yaml --mdl_config configs/model.yaml
python -m evaluation.run_eval --config configs/task3.yaml --mdl_config configs/model.yaml
python -m evaluation.run_eval --config configs/task4.yaml --mdl_config configs/model.yaml
```

The launcher uses `eval_mode` in each task config:

```yaml
eval_mode: inference   # uses the inference section
# or
eval_mode: post_val    # uses the post_val section
```

---

## 6. Configuration Files

### `configs/model.yaml`

Defines the base model and LoRA settings.

Main fields:

| Field | Meaning |
|---|---|
| `model.path` | Local Hugging Face model path. |
| `model.trust_remote_code` | Whether to trust custom tokenizer/model code. |
| `model.local_files_only` | Whether to avoid downloading from the internet. |
| `model.dtype` | Model dtype, e.g. `float16`, `bfloat16`, `float32`. |
| `lora.r` | LoRA rank. |
| `lora.alpha` | LoRA scaling factor. |
| `lora.dropout` | LoRA dropout. |
| `lora.target_modules` | Target model modules for LoRA injection. |

---

### `configs/preprocess.yaml`

Controls raw data input/output paths and enabled task generation.

Main fields:

| Field | Meaning |
|---|---|
| `paths.raw_data_dir` | Directory containing raw split JSON files. |
| `paths.processed_data_dir` | Directory where processed JSONL files are written. |
| `files.train/val/test` | Raw file names for each split. |
| `output.*_dir` | Output subfolders for each task. |
| `tasks.enable_*` | Whether to export each task. |

---

### `configs/task1.yaml` to `configs/task4.yaml`

Each task config controls training, prompt selection, data paths, output paths, labels, and evaluation mode.

Important fields:

| Field | Meaning |
|---|---|
| `task_name` | One of `user_emotion`, `therapist_strategy`, `therapist_emotion`, `therapist_response`. |
| `task_type` | `classification` or `generation`. |
| `train_path` | Training JSONL path. |
| `val_path` | Validation JSONL path. |
| `prompt_dir` | Directory containing prompt templates. |
| `output_dir` | Training output directory. |
| `init_adapter_path` | Optional adapter path for chained training. |
| `formatting.sph_arng_mod` | Dialogue modality rendering mode, e.g. `paired` or `field`. |
| `formatting.sys_reaction` | Task 4 output mode, e.g. `verbal` or `verbal_and_physical`. |
| `post_val` | Post-validation runtime config. |
| `inference` | Inference/ablation runtime config. |
| `eval_mode` | Selects `post_val` or `inference`. |

---

## 7. What Each Script Does

### 7.1 Common utilities

| File | Purpose |
|---|---|
| `common/config_utils.py` | Loads YAML configs and prompt files, selects prompt templates, sets Qwen special tokens, applies token IDs to the model, provides `PCSPrinter` structured logging, and prints dataset debug samples. |
| `common/dataset.py` | Defines dataset classes and prompt rendering logic. Converts processed JSONL records into tokenized prompt-target examples with prompt tokens masked as `-100`. |

Key classes/functions:

| Name | Role |
|---|---|
| `BaseTaskDataset` | Base PyTorch dataset for all tasks. |
| `EmotionDataset` | Dataset for Tasks 1-3 classification-by-generation. |
| `ResponseGenerationDataset` | Dataset for Task 4 generation. |
| `render_input_by_task()` | Builds task-specific prompt input text. |
| `serialize_response_target()` | Converts Task 4 structured target into text format. |
| `setup_qwen_special_tokens()` | Sets Qwen EOS/PAD tokens safely. |
| `PCSPrinter` | Lightweight stage/info/debug/warn logger. |

---

### 7.2 Raw dialogue concatenation / merging

| File | Purpose |
|---|---|
| `concatenate/func.py` | Normalizes raw dialogue fragments, splits utterance/video cue text, merges consecutive same-speaker fragments, and exports merged-turn dialogue JSON. |

Important functions:

| Function | Role |
|---|---|
| `split_utterance_and_video()` | Splits mixed text into utterance content and video cue content. |
| `build_fragment_record()` | Converts one raw dialogue item into a normalized fragment. |
| `build_merged_turn()` | Merges fragments from the same speaker into one turn. |
| `process_dataset()` | Processes a raw dataset file and saves a merged-turn version. |

---

### 7.3 Preprocessing

| File | Purpose |
|---|---|
| `preprocess/preprocess_all.py` | Main preprocessing entry point. Reads raw split JSON files and writes task-wise JSONL files. |
| `preprocess/core.py` | Core logic for converting each dialogue into Task 1-4 samples. |
| `preprocess/utils.py` | JSON/JSONL I/O, whitespace normalization, text fragment joining, and preview export helpers. |

Important functions:

| Function | Role |
|---|---|
| `normalize_turn()` | Standardizes one dialogue turn. |
| `build_history()` | Builds dialogue history before the current user turn. |
| `build_task1_sample()` | Builds a user-emotion classification sample. |
| `build_task2_sample()` | Builds a therapist-strategy classification sample. |
| `build_task3_sample()` | Builds a therapist-emotion classification sample. |
| `build_task4_sample()` | Builds a therapist-response generation sample. |
| `process_split()` | Processes one train/val/test split into task-wise records. |
| `export_one_split()` | Writes task-wise JSONL files and preview files. |

---

### 7.4 Training

| File | Purpose |
|---|---|
| `training/train_classification.py` | Trains Tasks 1-3 using classification-by-generation. Saves best LoRA adapter by validation weighted F1. |
| `training/train_generation.py` | Trains Task 4 response generation. Saves best LoRA adapter by validation loss. |
| `training/trainer_utils.py` | Shared training and validation utilities. |

Important functions:

| Function | Role |
|---|---|
| `train_one_epoch()` | Runs one optimization epoch. |
| `validate_loss()` | Computes validation loss. |
| `validate_generation()` | Generates classification labels and computes accuracy/F1. |
| `parse_label()` | Maps decoded model text back to a valid label. |

Training output:

```text
runs/<task_name>/run_001/
├── best_adapter/
├── train.log
└── metrics.json
```

---

### 7.5 Evaluation launcher and loaders

| File | Purpose |
|---|---|
| `evaluation/run_eval.py` | Unified entry point for classification and generation evaluation/inference. |
| `evaluation/loaders.py` | Loads models, tokenizers, datasets, dataloaders, label names, and runtime config sections. |
| `evaluation/export_utils.py` | Saves JSON, JSONL, and creates output directories. |
| `evaluation/record_utils.py` | Extracts sample indices and truncates long text for preview/ranking files. |
| `evaluation/excel_rankings.py` | Writes Excel files that rank difficult/wrong samples for manual error analysis. |

Important functions:

| Function | Role |
|---|---|
| `resolve_task_type()` | Detects whether the config is classification or generation. |
| `run_classification_eval()` | Full Task 1-3 evaluation/inference pipeline. |
| `run_generation_eval()` | Full Task 4 evaluation/inference pipeline. |
| `get_eval_section()` | Selects `post_val` or `inference` config based on `eval_mode`. |
| `resolve_inference_model_path()` | Resolves base model and adapter path for inference. |
| `save_classification_rankings_excel()` | Saves classification error-analysis workbook. |
| `save_generation_rankings_excel()` | Saves generation metric-ranking workbook. |

---

### 7.6 Classification evaluation modules

| File | Purpose |
|---|---|
| `evaluation/classification/inference.py` | Runs closed-set classification inference by scoring/generating candidate labels. |
| `evaluation/classification/metrics.py` | Builds classification metrics, reports, confusion matrices, and label/prediction distributions. |
| `evaluation/classification/artifacts.py` | Converts raw classification outputs into saved prediction records and metrics artifacts. |

Important functions:

| Function | Role |
|---|---|
| `run_classification_inference()` | Runs model inference for Tasks 1-3. |
| `build_final_metrics()` | Computes accuracy, weighted F1, macro F1, and sample count. |
| `build_confusion_matrix_dict()` | Creates confusion matrix JSON. |
| `build_classification_prediction_records()` | Builds JSONL-ready classification prediction records. |
| `build_classification_artifacts()` | Packages metrics and predictions for export. |

Classification evaluation output:

```text
runs/<task_name>/<run_dir>/inference/ or post_val/
├── metrics.json
├── predictions.jsonl
└── classification_rankings.xlsx
```

---

### 7.7 Generation evaluation modules

| File | Purpose |
|---|---|
| `evaluation/generation/inference.py` | Generates Task 4 responses and computes teacher-forced NLL/PPL on the gold reference target. |
| `evaluation/generation/metrics.py` | Computes ROUGE-L, BLEU-2, optional BERTScore, mean values, and aggregate generation metrics. |
| `evaluation/generation/artifacts.py` | Builds generation prediction records and metrics artifacts. |

Important functions:

| Function | Role |
|---|---|
| `run_generation_inference()` | Generates predictions and records reference, input, NLL, and PPL. |
| `bleu2_score()` | Computes a simple BLEU-2 score using whitespace tokenization. |
| `rouge_l_f1()` | Computes ROUGE-L F1. |
| `compute_bertscore_values()` | Computes BERTScore F1 if `bert_score` is installed. |
| `build_generation_prediction_records()` | Builds JSONL-ready generation prediction records. |
| `build_generation_metrics()` | Aggregates generation metrics from prediction records. |

Generation evaluation output:

```text
runs/therapist_response/<run_dir>/inference/ or post_val/
├── metrics.json
├── predictions.jsonl
├── predictions.json
└── generation_rankings.xlsx
```

Note: PPL is currently computed during generation inference using teacher forcing on the gold target. This measures how surprised the model is by the reference response, not how good the generated text is. BLEU/ROUGE/BERTScore are text-comparison metrics computed from prediction/reference strings.

---

## 8. Data Format

Each processed JSONL record contains fields like:

```json
{
  "dialog_index": 0,
  "turn_index": 4,
  "task": "user_emotion",
  "problem_type": "...",
  "situation": "...",
  "history": [],
  "current_user_turn": {
    "utterance_text": "...",
    "video_text": "...",
    "utterance_list": ["..."],
    "video_list": ["..."]
  },
  "fields": {},
  "target": "neutral"
}
```

For Task 4, `target` is structured:

```json
{
  "target": {
    "utterance_list": ["..."],
    "video_list": ["..."]
  }
}
```

---

## 9. Ablation / Inference Controls

Task configs include an `inference.ablation` section.

Example:

```yaml
inference:
  ablation:
    video: true
    history: false
    emotion: true
    strategy: true
```

Typical meanings:

| Flag | Meaning |
|---|---|
| `video` | Whether video/audio textual cues are included. |
| `history` | Whether dialogue history is included. |
| `emotion` | Whether user emotion field is included for downstream tasks. |
| `strategy` | Whether therapist strategy field is included for response generation. |

Use this to test which component contributes to final performance.

---

## 10. environment settings
since the model applied here is a old model, the corresponding libraries used here are also the previous versions. Here are the versions of the third-party packages imported in here.

```text
Python 3.10.20
torch 2.5.1+cu121
transformers 4.32.0
peft 0.10.0
yaml 6.0.3
sklearn 1.7.2
openpyxl 3.1.5
bert_score 0.3.12
tiktoken 0.12.0
```