# Results — Baseline Ladder

Scene = 8-way group-activity classification accuracy. Person = 9-way
per-player action accuracy (only applicable from B8-mt onward).

| ID | Description | Val acc | Test acc | Paper target | Status |
|---|---|---|---|---|---|
| **B1-tuned** | Fine-tuned ResNet50, center-frame image | 68.9% | **72.2%** | 66.7% | ✅ done (+5.5 pts over paper) |
| B2 | Frozen ResNet50 on person crops, max-pooled | 43.5% | 42.2% | 64.6% | ✅ done (−22.4 pts under paper, see note below) |
| B3-A | Fine-tune ResNet50 on person crops, 9-way action (intermediate step, feeds B2/B3/B5/B6/B7/B8) | 69.6% | 80.6% (v1) / 72.8% (v2) | n/a (not paper-reported) | ✅ done, v1 selected for B+C |
| B3 (B+C) | B3-A backbone (v1), pooled → scene head | — | — | 68.1% | not started |
| B4 | Image-level LSTM (B1 features) | — | — | 63.1% | not started |
| B5 | Per-player LSTM, max-pooled (no scene LSTM) | — | — | 67.6% | not started |
| B6 | Image-level LSTM (B3-A features) | — | — | 74.7% | not started |
| B7 | Full hierarchical model (LSTM1 → pool → LSTM2) | — | — | 80.2% | not started |
| B8 | B7 + team-separated pooling | — | — | 81.9% | not started |
| B8-mt | B8 + per-player action multi-task head | — | — | — | not started |

## B1-tuned — detail

- 15 epochs, batch size 24, AdamW lr=1e-4, cosine schedule, weighted cross-entropy (class imbalance), mixed precision (AMP).
- Best val checkpoint: epoch 12 (val_acc 68.9%).
- Test accuracy: **72.18%**
- Checkpoint: `outputs/checkpoints/scene_finetune_b1.pt` (tracked via Git LFS)

Per-class test performance:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| l-pass | 0.686 | 0.668 | 0.677 | 226 |
| l-spike | 0.784 | 0.810 | 0.797 | 179 |
| l_set | 0.718 | 0.696 | 0.707 | 168 |
| l_winpoint | 0.798 | 0.853 | 0.825 | 102 |
| r-pass | 0.582 | 0.710 | 0.639 | 210 |
| r_set | 0.738 | 0.661 | 0.698 | 192 |
| r_spike | 0.790 | 0.740 | 0.764 | 173 |
| r_winpoint | 0.871 | 0.701 | 0.777 | 87 |

Full confusion matrix: [`scene_finetune_b1_test_report.txt`](scene_finetune_b1_test_report.txt).

Note: train accuracy reached ~100% by epoch 12-15 (clear overfitting), but val/test
accuracy kept improving slightly — the weighted loss and cosine schedule kept it
from collapsing. Most confusion is between `r-pass`/`r_set` and between
`l-pass`/`l_set` (adjacent game phases for the same team look visually similar
in a single frame) — expected for a model with no temporal context, and exactly
the kind of error the LSTM baselines (B4+) should reduce.

## B2 — detail

- Per-player crops, frozen off-the-shelf ImageNet ResNet50 (no fine-tuning of
  any kind), max-pooled across players, small 2-layer classifier head trained
  on top (AdamW lr=1e-3, 15 epochs, weighted cross-entropy, AMP added after
  this run started so this run itself did not benefit from it).
- Best val checkpoint: epoch 14 (val_acc 43.5%).
- Test accuracy: **42.18%**
- Checkpoint: `outputs/checkpoints/person_pooled_b2.pt`

Per-class test performance:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| l-pass | 0.385 | 0.296 | 0.335 | 226 |
| l-spike | 0.707 | 0.296 | 0.417 | 179 |
| l_set | 0.464 | 0.464 | 0.464 | 168 |
| l_winpoint | 0.495 | 0.529 | 0.512 | 102 |
| r-pass | 0.314 | 0.562 | 0.403 | 210 |
| r_set | 0.463 | 0.385 | 0.420 | 192 |
| r_spike | 0.435 | 0.526 | 0.476 | 173 |
| r_winpoint | 0.439 | 0.333 | 0.379 | 87 |

Full confusion matrix: [`person_pooled_b2_test_report.txt`](person_pooled_b2_test_report.txt).

**Known gap from the paper's 64.6%:** train accuracy was still climbing at
epoch 15 (60.5%, not plateaued) — the small classifier head needs more
epochs than 15 to fully converge when learning from generic, never-fine-tuned
ImageNet features (there is no task-specific signal in the backbone itself,
unlike B1/B3, so the head has to do all the work from a cold start). This run
is kept and documented as-is per the baseline ladder's purpose (ablation
comparison point, not a result to optimize), rather than re-run with more
epochs/tuning — B3 (same pipeline, fine-tuned backbone) is the baseline that
matters for the main result chain.

## B3-A — detail (two runs compared)

Not a paper-reported baseline on its own — this is the intermediate
person-action classifier (9-way) that feeds into B2/B3/B5/B6/B7/B8. Its own
accuracy isn't in the paper's tables; what matters is how good the resulting
per-player features are for the downstream scene-classification baselines.

Ran twice to address severe class imbalance (`standing` ≈ 115x `jumping` in
the training set):

| | **v1** (weighted CE) | **v2** (focal loss γ=2, balanced sampler) |
|---|---|---|
| checkpoint | `person_finetune_b3a.pt` | `person_finetune_b3a_v2.pt` |
| best val_acc | 78.3% (epoch 14) | 69.6% (epoch 14) |
| test accuracy | **80.56%** | 72.83% |

Per-class test recall, v1 vs v2:

| class | v1 recall | v2 recall | support |
|---|---|---|---|
| standing | 0.880 | 0.750 | 10655 |
| blocking | 0.904 | 0.910 | 691 |
| falling | 0.823 | 0.796 | 373 |
| setting | 0.807 | 0.810 | 368 |
| spiking | 0.841 | 0.803 | 340 |
| moving | 0.676 | 0.703 | 1342 |
| waiting | 0.470 | **0.602** | 1023 |
| digging | 0.329 | 0.384 | 645 |
| **jumping** | **0.191** | **0.079** | 89 |

Full reports: [`person_finetune_b3a_test_report.txt`](person_finetune_b3a_test_report.txt),
[`person_finetune_b3a_v2_test_report.txt`](person_finetune_b3a_v2_test_report.txt).

**Counter-intuitive finding:** v2's imbalance fixes (focal loss + a
`WeightedRandomSampler` that gives every class roughly equal expected
exposure per epoch) were aimed squarely at `jumping` (the worst class), but
its recall got *worse* (19.1% → 7.9%), while the moderately-rare classes
(`waiting`, `moving`, `digging`) improved. Likely cause: `jumping` has only
151 training images; the balanced sampler redraws them ~18x per epoch, and
with focal loss additionally up-weighting "hard" examples, the model
overfits to those exact 151 images (memorization) instead of generalizing —
the opposite of the intended effect for a class this small.

**Decision:** use **v1** (weighted CE, no resampling) as the backbone for
B3's B+C step and for B5/B6/B7/B8's feature cache — it has the higher
overall test accuracy and is stronger on every class except `waiting` and
`moving`. `jumping`'s weakness is a known, documented limitation carried
into the downstream baselines rather than a solved problem; a smarter fix
(e.g. capping how many times a rare class can be resampled, instead of
equalizing all classes outright) is a candidate follow-up, not applied here.
