# Standalone retinal review-priority baseline

## Scope and semantics

This experiment is separate from RetinaReady's capture-quality pipeline and UI.
It ranks retinal photographs for human review and returns one of:

- `ROUTINE`: lower review priority; **not** a finding that disease is absent.
- `PRIORITY`: higher review priority; **not** a diagnosis or treatment advice.
- `UNCERTAIN`: abstain and route for human review.

DeepDRiD eye grades `0-1` provide the research `ROUTINE` truth and grades
`2-4` provide the research `PRIORITY` truth. `UNCERTAIN` is a calibrated model
abstention, never an invented training label. This mapping is a prototype
workflow convention, not a clinically validated urgency taxonomy.

## Data, split, and license audit

Only local DeepDRiD v1.1 regular-fundus training and validation data are used.
The dataset's included license is CC BY-SA 4.0; preserve attribution, indicate
modifications, and share adapted dataset material alike. Cite Liu et al.,
“DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation
Challenge,” *Patterns* 3 (2022), DOI `10.1016/j.patter.2022.100512`.

`ml/prepare_escalation_manifests.py` deterministically stratifies official
training patients by patient maximum eye grade (seed 42):

| Manifest | Source | Patients | Images | Role |
| --- | --- | ---: | ---: | --- |
| `train.csv` | official training | 150 | 600 | fit |
| `val.csv` | official training | 30 | 120 | epoch selection |
| `calibration.csv` | official training | 120 | 480 | thresholds only |
| `eval.csv` | official validation | 100 | 400 | final evaluation only |

Every pair of partitions has zero patient overlap. DeepDRiD test, MSHF, and UWF
data are refused by both scripts. Each generated row includes `dr_grade`,
`escalation_label`, `overall_quality`, and `source_split`. Eleven official
training filenames have a left/right suffix that disagrees with the populated
eye-grade column; truth is therefore derived from the one non-empty
`left_eye_DR_Level` or `right_eye_DR_Level` field, and the discrepancy is
retained in `filename_side_matches_grade_field`.

## Model and conservative policy

The baseline uses cached 1024-dimensional global features from a frozen
ImageNet DenseNet-121 and a five-member, 64-hidden-unit MLP ensemble. It is
small and standalone; no Gemma weights or deployed quality artifacts are
changed.

Epochs are selected only on internal `val`, then each member is refit on
`train+val`. Patient-grouped calibration controls two adverse events with
strict thresholds and one-sided exact binomial bounds:

- any `PRIORITY`-truth image for a patient called `ROUTINE`;
- any `ROUTINE`-truth image for a patient called `PRIORITY`.

At 5% per-gate nominal risk and delta 0.05, calibration allowed zero errors.
The thresholds are `score < 0.000060109 -> ROUTINE` and
`score > 0.999606431 -> PRIORITY`; everything else is `UNCERTAIN`. The joint
confidence lower bound is 90%. These historical calibration bounds are not a
deployment guarantee.

## Official-validation results

The official validation set was not used for model or threshold selection.

| Metric | Result |
| --- | ---: |
| Image ROC-AUC | 0.92997 |
| Patient ROC-AUC | 0.95160 |
| Image coverage | 38.75% (155/400) |
| Accepted-image accuracy | 97.42% (151/155) |
| False `ROUTINE` among `PRIORITY` images | 1.67% (3/180); exact upper 95% 4.25% |
| False `PRIORITY` among `ROUTINE` images | 0.45% (1/220); exact upper 95% 2.14% |
| Patient adverse false-`ROUTINE` events | 4.0% (2/50); exact upper 95% 12.06% |
| Patient aggregate coverage | 37% (37/100) |
| Patient aggregate accepted accuracy | 97.30% (36/37) |
| Patient aggregate false `ROUTINE` | 0/50; exact upper 95% 5.82% |

The model abstained on 245/400 images. On technically adequate
(`overall_quality=1`) images it made zero accepted errors at 39.01% coverage;
all four accepted image errors occurred in the `overall_quality=0` stratum.
That is descriptive only and does not connect this experiment to the deployed
quality gate.

## Reproduction

```bash
python3 ml/prepare_escalation_manifests.py
python3 ml/experiment_escalation_baseline.py --device cpu
python3 -m unittest discover -s ml/tests -v
```

Primary artifacts:

- `data/escalation-manifests/summary.json`
- `outputs/escalation-baseline/report.json`
- `outputs/escalation-baseline/escalation-baseline-experimental.pt`

The model artifact explicitly sets `runtime_integration_authorized=false` and
`diagnostic_use_authorized=false`.

## Opted-in hackathon runtime adapter

The source artifact and the recommendation above remain unchanged. For the
local, nonclinical hackathon research demo only, a separate exact-hash
promotion manifest may activate `app/escalation_specialist.py`:

```bash
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_DEVICE=cpu
```

`models/retinaready-escalation-demo/promotion-manifest.json` binds the exact
artifact, report, and DenseNet backbone. The adapter re-verifies all three
before every release attempt, uses the training preprocessing, stored feature
normalization, five frozen heads, and exact strict thresholds, and maps
`ROUTINE` / `PRIORITY` to `ROUTINE_REVIEW` / `PRIORITY_REVIEW`. Opt-out,
integrity drift, threshold equality, the abstention interval, decode errors,
and inference errors all return `UNCERTAIN`. This narrow demo allowlist is not
clinical promotion or deployment authorization.

## Limitations and next evidence

This is a single-dataset retrospective baseline from diabetes-screening
cohorts, not clinical validation. Eye labels repeat across correlated dual
views; patient grouping prevents split leakage but does not create independent
images. Frozen ImageNet features may exploit acquisition or quality shortcuts.
The 100-patient evaluation has wide patient-level bounds and has been used
elsewhere in this project for technical-quality research, so it is untouched
in this experiment but not project-level fresh.

Do not integrate it. Next evidence should include clinician review of the
priority mapping, an external multi-device patient-disjoint evaluation, fresh
calibration, and prospective workflow testing.
