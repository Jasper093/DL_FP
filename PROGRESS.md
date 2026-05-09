# What’s still missing

## TL;DR

Your current progress by experiment group:

```
Group A: Faithful 4-task reproduction        ✅ Mostly done, but not fully end-to-end
Group B: Direct generation vs structured     ❌ Not done yet
Group C: Ablation on intermediate tasks      ✅ Mostly done, not yet analyzed
Group D: Oracle analysis                     ❌ Not done yet
```

---
# Progress by experiment group

## Overall map

```
Proposal experiment groups
────────────────────────────────────────────
Group A: Faithful reproduction
    ├── Preprocessing                      ✅ Done
    ├── 4 task datasets                    ✅ Done
    ├── Task-wise training                 ✅ Done
    ├── Task-wise post-validation          ✅ Done
    └── True predicted-label pipeline      ❌ Missing

Group B: Direct vs structured reasoning
    ├── Structured model                   🟡 Partial
    └── Direct response baseline           ❌ Missing

Group C: Ablation
    ├── Remove emotion                     ✅ Done
    ├── Remove strategy                    ✅ Done
    └── Remove multimodal cues             ✅ Done

Group D: Oracle analysis
    ├── gold_all                           ✅ Done
    ├── pred_all                           ❌ Missing
    ├── oracle_emotion                     ❌ Missing
    ├── oracle_strategy                    ❌ Missing
    └── oracle_system_emotion              ❌ Missing
```

---

# Group A — Faithful reproduction of the full pipeline

The proposal says Group A should reproduce the full 4-task reasoning model and verify that it can generate outputs in this order:

```
user emotion → strategy → system emotion → response
```

The proposal’s success criterion is stable training and plausible intermediate predictions, not necessarily matching the original paper’s exact numbers.

## Group A status: ✅ Mostly done / 🟡 Partial

| Item | Status | Your current progress |
| --- | --- | --- |
| MESC task-wise data preprocessing | ✅ Done | You have processed JSONL files for the four tasks. |
| Task 1: user emotion recognition | ✅ Done | Training/evaluation framework exists. |
| Task 2: therapist strategy prediction | ✅ Done / Partial | Training/evaluation exists, but likely uses gold previous labels. |
| Task 3: system/therapist emotion prediction | ✅ Done / Partial | Training/evaluation exists, but likely uses gold previous labels. |
| Task 4: response generation | ✅ Done / Partial | Generation training/evaluation exists, but likely uses gold intermediate labels. |
| Task-specific prompts | ✅ Done | You have separate prompts for the four tasks. |
| YAML config system | ✅ Done | You have config-driven task execution. |
| LoRA adapter training | ✅ Done | Task-wise adapter training exists. |
| Post-validation | ✅ Done | Metrics and prediction outputs exist. |
| Error analysis artifacts | ✅ Done | Ranking Excel / confusion matrix / prediction artifacts exist. |
| True full pipeline inference | ❌ Missing | Task2 does not yet consume Task1 prediction, Task3 does not consume previous predictions, Task4 does not consume all predicted labels. |

## What is still missing for Group A

You still need:

```
run_pipeline_inference.py

Task1 prediction
  ↓
insert into Task2 input
  ↓
Task2 prediction
  ↓
insert into Task3 input
  ↓
Task3 prediction
  ↓
insert into Task4 input
  ↓
final response generation
```

This is the main missing bridge.

---

# Group B — Direct generation vs structured reasoning

The proposal says Group B compares the reproduced 4-task reasoning model against a **direct response-generation baseline**. This is described as the main scientific comparison.

## Group B status: ❌ Not done yet

| Item | Status | Explanation |
| --- | --- | --- |
| Structured 4-task model | 🟡 Partial | You have task-wise structured training, but not full predicted-label inference. |
| Direct response-generation baseline | ❌ Missing | You need a model/config that generates therapist response directly without emotion/strategy/system-emotion fields. |
| Comparison table | ❌ Missing | No final table comparing direct vs structured. |
| Interpretation of result | ❌ Missing | Cannot write conclusion until direct baseline exists. |

## What direct baseline should look like

```
Input:
  problem_type
  situation
  dialogue history
  current user utterance
  video/textual cues if used

No input:
  user_emotion
  therapist_strategy
  therapist_emotion

Output:
  therapist response
```
---

# Group C — Ablation on intermediate tasks

The proposal says Group C should remove one intermediate sub-task at a time, especially:

```
-emotion
-strategy
-multimodal cues
```

to test which part contributes most to final response quality.

## Group C status: ✅ Done

| Ablation | Status | Your current progress |
| --- | --- | --- |
| Remove user emotion ✅ Done | Drop-emotion style configs/runs is constructed.|
| Remove strategy | ✅ Done | Drop-strategy style configs/runs is finished. |
| Remove multimodal cues/video | ✅ Done | drop-video / drop-cue style settings are all set. |
| Remove history | ✅ Done | Useful, but not explicitly one of the main proposal ablations. |

## What Group C still needs

```
1. Confirm each ablation config actually runs.
2. Run each ablation on the same validation/test split.
3. Save metrics consistently.
4. Make one table:

Full model
-emotion
-strategy
-multimodal cues
-direct baseline if included
```

The original SMES paper also reports ablations for removing emotion and strategy, using metrics like PPL, BLEU-2, BLEU-4, and ROUGE-L.

---

# Group D — Oracle analysis

The proposal says Group D should replace predicted intermediate labels with gold labels during inference to measure error propagation. It specifically asks whether gold emotion or gold strategy improves final response metrics more.

## Group D status: ❌ Not done yet

| Oracle mode | Status | Meaning |
| --- | --- | --- |
| `gold_all` | ✅ Done | Task4 uses gold emotion + gold strategy + gold system emotion. |
| `pred_all` | ❌ Missing | Task4 uses predicted emotion + predicted strategy + predicted system emotion. |
| `oracle_emotion` | ❌ Missing | Replace only emotion with gold label. |
| `oracle_strategy` | ❌ Missing | Replace only strategy with gold label. |
| `oracle_system_emotion` | ❌ Missing | Replace only system emotion with gold label. |
| Error propagation analysis | ❌ Missing | Need to compare metric differences across oracle modes. |

Important dependency:

```
Group D depends on Group A end-to-end inference.
```

You cannot do real oracle analysis until you have:

```
pred_all = Task1 pred → Task2 pred → Task3 pred → Task4 response
```

---

# Metrics status

The proposal says you planned these metrics:

| Task | Proposed metrics | Current status |
| --- | --- | --- |
| User emotion | Accuracy, weighted F1 | ✅ Done |
| Strategy | Accuracy, weighted F1 | ✅ Done |
| System emotion | Accuracy, weighted F1 | ✅ Done |
| Response generation | BLEU-2, BLEU-4, ROUGE-L, BERTScore | 🟡 Mostly done |
| Human evaluation | Optional | ❌ Not done |

The proposal explicitly includes BLEU-2, BLEU-4, ROUGE-L, and BERTScore for response generation.

Main missing metric:

```
BLEU-4
```

---

# Priority list from here

```
1. Finish Group A missing part
   end-to-end predicted-label inference

2. Finish Group B
   direct response-generation baseline

3. Finish Group D
   oracle analysis modes
```