# Volleyball Group Activity Recognition

Reproduction and extension of **"A Hierarchical Deep Temporal Model for Group
Activity Recognition"** (Ibrahim et al., CVPR 2016) on the official Volleyball
dataset, implemented fresh in PyTorch.

## Task

Given a volleyball video clip, predict:
1. **Group/scene activity** (8 classes) — what the team as a whole is doing
   (e.g. `r_spike`, `l-pass`, `r_winpoint`).
2. **Per-player action** (9 classes) — what each individual player is doing
   (e.g. `standing`, `spiking`, `blocking`) — a mandatory extension beyond
   the original paper (see Section 6 of the project brief).

## Dataset

~4,830 annotated clips across 55 videos. Each clip is a 9-frame window
centered on an annotated frame, with ground-truth scene label and per-player
bounding boxes + action labels. Official train/val/test split is by video ID
(24/15/16 videos) — see [`src/data/annotations.py`](src/data/annotations.py).

Dataset files are not pushed to this repo (too large, ~60GB).

- **Download**: [Google Drive folder](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing)
- **Expected local layout**: see `src/data/annotations.py`

## B1-tuned — explained

The first and simplest baseline: a single whole-clip image goes in, one
label for the *whole scene* comes out. No individual players, no bounding
boxes, no time dimension.

![B1 pipeline](outputs/results/b1_pipeline.png)

ResNet50 (ImageNet-pretrained) is fine-tuned end-to-end: the center frame of
each clip is fed through the backbone, globally pooled to a 2048-d vector,
and passed through a linear classifier over the 8 scene classes.

**Result: 72.2% test accuracy**, beating the paper's reported 66.7% for this
exact baseline.

![B1 training curve](outputs/results/b1_training_curve.png)

Train accuracy reaches ~100% by epoch 12 (clear overfitting on only 3,864
training images), but validation/test accuracy keeps climbing slightly
thanks to the weighted loss (corrects for class imbalance) and cosine LR
schedule — it doesn't collapse the way an unweighted/uncontrolled run would.

![B1 confusion matrix](outputs/results/b1_confusion_matrix.png)

Reading the confusion matrix: almost all the error is between game phases of
the *same* team — `r-pass` gets confused with `r_set` (27 cases), `l-pass`
with `l_set` (32 cases) — because a single static frame often can't tell a
reception from a set-up without seeing the motion. This is exactly the
failure mode later baselines (B4+, which add an LSTM over the 9-frame
sequence) are designed to fix.

## Baseline ladder

Reproducing the paper's ablation study, from simplest to the full model:

| ID | Description | Paper target (test acc) |
|---|---|---|
| B1-tuned | Fine-tuned ResNet50 on center-frame image, 8-way scene classification | 66.7% |
| B2 | Frozen ResNet50 feature extractor (comparison) | 64.6% |
| B3 | Fine-tuned ResNet50 on person crops (9-way action) + max-pool → scene head | 68.1% |
| B4 | Image-level LSTM over 9 frames (B1 features) | 63.1% |
| B5 | Per-player LSTM (no scene-level LSTM), max-pooled | 67.6% |
| B6 | Image-level LSTM over B3-A-derived features | 74.7% |
| B7 | Full hierarchical model: LSTM1 (player) → max-pool → LSTM2 (scene) | 80.2% |
| B8 | B7 + team-separated pooling (group-style pooling) | 81.9% |
| B8-mt | B8 + per-player action head, multi-task loss | — |

Live results table: [`outputs/results/RESULTS.md`](outputs/results/RESULTS.md).

## Repo structure

```
src/
  data/         dataset loading, splits, PyTorch Dataset classes
  models/       ResNet50 backbone, LSTM1/LSTM2/pooling, classification heads
  training/     Phase A (backbone fine-tuning), Phase B (feature caching),
                Phase C (temporal models B4-B8)
  evaluation/   accuracy / confusion matrix / per-class precision-recall-F1
  inference/    video -> timestamped event timeline (later stage)
  ball/         ball-presence detection (later stage, best-effort)
outputs/
  checkpoints/  trained model weights (tracked via Git LFS)
  results/      per-baseline metrics (JSON + text report) and RESULTS.md
  features_cache/  cached backbone features for Phase C (not committed, large)
configs/        per-baseline hyperparameter configs (as needed)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install tqdm
```

## Training a baseline

```bash
# Phase A: fine-tune backbone (B1 / B2 / B3-A)
.venv\Scripts\python.exe -m src.training.train_backbone --task scene --mode finetune --epochs 15 --batch-size 24 --name scene_finetune_b1

# Phase B: cache features from a trained backbone (needed for B4-B8)
.venv\Scripts\python.exe -m src.training.cache_features --level image --checkpoint outputs/checkpoints/scene_finetune_b1.pt --name b1_image

# Phase C: train a temporal model on cached features (B4-B8)
.venv\Scripts\python.exe -m src.training.train_temporal --variant b7 --feature-name b3a_person
```

## Status

See [`outputs/results/RESULTS.md`](outputs/results/RESULTS.md) for current progress against the baseline ladder.
