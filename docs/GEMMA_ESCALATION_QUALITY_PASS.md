# Gemma escalation: quality-pass-only lineage

The review-priority model is defined only for clinically usable conventional
color fundus photographs. `ROUTINE` and `PRIORITY` are therefore trained,
calibrated, and evaluated only after the independent capture-quality gate has
passed. They are routing labels, not diagnoses or claims that an eye is healthy.

## Deterministic derivation

Run from the repository root:

```bash
python3 ml/prepare_escalation_quality_pass_manifests.py
```

The generator validates all four existing patient-disjoint escalation
partitions and their recorded hashes before writing anything. It refuses bad
quality values, grade/label mismatches, blank identifiers, missing or empty
image files, duplicate image IDs, split mismatches, and patient overlap. It
then performs one operation only: retain rows whose official DeepDRiD
`overall_quality` value is `1`. Column order, row order, labels, image paths,
and split assignments are preserved.

Inputs remain in `data/escalation-manifests/`. Derived artifacts are written
to `data/escalation-quality-pass-manifests/`; the parent data is not modified.
`summary.json` records the generator hash, parent-summary hash, every parent
and derived manifest hash, counts, license lineage, and pairwise patient/image
overlap audits.

| Partition | Images | Patients | ROUTINE | PRIORITY | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 294 | 101 | 154 | 140 | `4027f635cedb88c25c28fbf14f78184980ee6fc9e23ab27c4af95e50ea82ecac` |
| val | 70 | 25 | 40 | 30 | `a419ddd601c51f3b02976c512105dafc7134ab654ca9a21e3ffaeb8a0d0610b3` |
| calibration | 212 | 70 | 114 | 98 | `88c813baad4339a09f63459c1c342f9c5d7420e138ab5fb9771a2749fc1a9db8` |
| eval | 182 | 65 | 94 | 88 | `4392a038087ad44adceae6d177d9616814201dec8356b993eefd4bd098a9c099` |

All pairwise patient-ID and image-ID overlaps are zero. The source contains
1,600 images; 758 pass quality and 842 are excluded. Parent summary SHA-256 is
`66de96b85c14558d12637b319b35f8f809b06a473f9c149c6f1cabeff8168b61`;
derived summary SHA-256 is
`58f908fda63a0e41c931bcbb0c782a065ea1fd36a9d7b441103cc4e9bb095534`.

## Preflight and training

The new full-run config is
`ml/configs/gemma4_26b_escalation_quality_pass_full.json`. It declares all
four filtered manifests and writes to a new run directory, leaving the legacy
config and prior outputs untouched.

```bash
python3 ml/preflight_a100.py \
  --config ml/configs/gemma4_26b_escalation_quality_pass_full.json \
  --json-output ml/runs/a100-escalation-quality-pass-preflight.json

python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_escalation_quality_pass_full.json \
  --dry-run
```

Training and candidate selection are complete. Training consumed only train
and validation; calibration and evaluation were not consulted until after the
winner was frozen. Unusable images remain the upstream quality gate's
responsibility and must never be silently assigned `ROUTINE`.

## Validation-only selection

All candidates were ranked first by priority-positive ROC-AUC on the same
complete 70-image quality-pass validation manifest:

| Candidate | AUC | Balanced accuracy | Priority recall | Routine recall |
| --- | ---: | ---: | ---: | ---: |
| q/v rank-16 step-60 adapter | **0.92375** | **0.83333** | **0.86667** | 0.80000 |
| 256-image q/v challenger | 0.89500 | 0.79583 | 0.76667 | **0.82500** |
| q/k/v/o full-run root | 0.88000 | — | — | — |
| q/k/v/o checkpoint 74 | 0.868333 | — | — | — |
| q/k/v/o checkpoint 111 | 0.88000 | 0.741667 | 0.633333 | 0.85000 |

The q/v step-60 adapter won on the declared primary metric and also had the
highest priority recall. Its PEFT weights SHA-256 is
`ba22230a8e49e7281c4ca5c2d56886dd4e22c14d25aa710c8f4f5af6168f7e2e`.

## Frozen evaluation

The selected adapter's direct decision logits produced:

| Partition | Images | Patients | Priority AUC | Balanced accuracy at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| calibration | 212 | 70 | 0.938238 | — |
| evaluation | 182 | 65 | **0.956903** | **0.912355** |

The calibration-frozen selective policy releases `ROUTINE` only for scores
strictly below `0.0002611903190957194`, releases `PRIORITY` only for scores
strictly above `0.9993736658418905`, and returns `UNCERTAIN` otherwise. On the
evaluation partition it accepted 65/182 images (35.7% coverage): 36 Priority
and 29 Routine. All 65 accepted routes were correct; 117 abstained.

This policy is research evidence for a 10% patient-event risk target per error
type with 90% simultaneous confidence. The calibration cohort is too small to
certify a 5% target. It is not clinical validation.

The live llama.cpp demo uses constrained free generation from the converted
LoRA and is deliberately labeled uncalibrated. It demonstrates the trained
Gemma task and fail-closed application contract but does not inherit the
direct-logit thresholds or their confidence claims.
