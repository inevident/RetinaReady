# Offline combined quality-gate + review-priority evaluation

## Scope

`ml/evaluate_combined_pipeline.py` is the reproducible specialist-only offline
policy-composition evaluator for the frozen RetinaReady quality specialist
followed by the experimental RetinaPriority baseline. It is not an upload-API
or hybrid Gemma-veto runtime evaluation. It runs no new model inference, uses
no cloud resource, and does not change either component policy or any
application runtime.

This is retrospective research evidence on the repeatedly viewed 400-image,
100-patient DeepDRiD v1.1 official validation cohort. It is not a fresh test,
clinical validation, diagnosis, disease exclusion, urgency taxonomy, or
treatment recommendation. `ROUTINE_REVIEW` means only lower queue priority; it
does not mean healthy or no disease. `PRIORITY_REVIEW` means only higher queue
priority under the experimental grade-derived research convention.

## Pinned inputs and policy fidelity

The evaluator starts from
`ml/configs/combined_offline_evaluation_v1.json` and fails before evaluation if
any of its 11 local inputs has a different SHA-256. The lock covers:

- the frozen quality bundle manifest, DenseNet-121 backbone, both specialist
  heads, technical-quality truth manifest, and 400-row score report;
- the prior full hybrid report that independently records the quality report's
  SHA-256, so the reused report has a pre-existing hash attestation;
- the escalation truth manifest, 400-row score report, model artifact, and
  research-demo promotion manifest that binds the report, artifact, backbone,
  fail-closed behavior, and exact thresholds.

The canonical quality source is
`outputs/quality-specialist-rigorous-factors/report.json`, not the auxiliary
weight-zero decision challenger. It contains 400 image-level `decision_score`
rows and is the source report named by the bundled quality heads.

The evaluator recomputes decisions from scores instead of trusting stored
labels:

- quality score `> 0.9498774409294128` becomes `READY`;
- quality score `< 0.015006430447101593` becomes `RETAKE`;
- equality and everything between becomes `LIMITED`;
- review-priority score `< 0.00006010895594954491` becomes internal
  `ROUTINE`;
- review-priority score `> 0.9996064305305481` becomes internal `PRIORITY`;
- equality and everything between becomes `UNCERTAIN`.

Those recomputed standalone distributions must exactly reproduce the two
source reports: quality `21 READY / 23 RETAKE / 356 LIMITED`, and escalation
`99 ROUTINE / 56 PRIORITY / 245 UNCERTAIN`. A disagreement aborts rather than
silently changing policy.

## Join and truth contract

The join key is `image_id`; ordering is never used. All four sources must have
exactly the same 400 unique IDs. For every ID, `patient_id`, `image_path`, and
`source_split` must match. Technical-quality truth must agree between both
manifests and both reports. DR grade must agree between the escalation manifest
and report.

Technical-quality truth is `overall_quality=1 -> READY` and
`overall_quality=0 -> RETAKE`. DR grade truth comes from the single populated
DeepDRiD left/right grade field retained in the escalation manifest, never from
the filename side. Grades `0-1` map to research `ROUTINE` truth and grades
`2-4` map to research `PRIORITY` truth. `UNCERTAIN` is an abstention, never a
truth label.

Any duplicate, missing, extra, malformed, non-finite, out-of-range, or
cross-source-mismatched row fails closed. The official test split, MSHF, UWF,
and alternative external model families are not inputs.

## Quality-first pipeline

Only exact quality `READY` executes the simulated review-priority stage.
Quality `RETAKE` and `LIMITED` are final blocked states. For audit only, the
standalone escalation report supplies a cached score for every image; a
blocked row explicitly records
`review_priority_stage_executed_in_simulated_pipeline=false`, and its cached
score never affects the final state.

The five image-level final states are:

- `RETAKE`: quality gate requests recapture;
- `LIMITED`: quality gate abstains/limits use;
- `ROUTINE_REVIEW`: quality passed and the escalation score was decisively
  routine under the research policy;
- `PRIORITY_REVIEW`: quality passed and the escalation score was decisively
  priority under the research policy;
- `UNCERTAIN`: quality passed but the escalation policy abstained.

## Full 400-image result

Every image remains in the full-run denominator.

| Final pipeline state | Images | All 400 |
| --- | ---: | ---: |
| `RETAKE` | 23 | 5.75% |
| `LIMITED` | 356 | 89.00% |
| `ROUTINE_REVIEW` | 4 | 1.00% |
| `PRIORITY_REVIEW` | 5 | 1.25% |
| `UNCERTAIN` | 12 | 3.00% |
| **Total** | **400** | **100%** |

Quality blocked 379/400 images (94.75%). Twelve more images were downstream
`UNCERTAIN`, so 391/400 (97.75%) received no decisive review-queue release.
Strict abstention (`LIMITED` plus `UNCERTAIN`) was 368/400 (92.0%); the other
23 quality-blocked images were actionable `RETAKE` calls, not abstentions.
Full-cohort decisive review coverage was 9/400 (2.25%), or 9/21 (42.86%) among
quality-`READY` images. All 9 selected releases matched the grade-derived
review truth, but 9/9 is a tiny selected subset and is not general performance
evidence.

Within the 21-image quality-passed subset, truth was 13 `PRIORITY` and 8
`ROUTINE`; decisions were 5 `PRIORITY`, 4 `ROUTINE`, and 12 `UNCERTAIN`.
Conditional false releases were 0/13 and 0/8. These conditional denominators
are reported separately and never replace the full 400-image metrics.

Technical-quality truth was 182 `READY` and 218 `RETAKE`. The standalone
quality gate produced 2 false `READY` calls among 218 true `RETAKE` images
(0.92%) and 1 false `RETAKE` among 182 true `READY` images (0.55%). Its
selective coverage was 44/400 (11.0%) and accepted accuracy was 41/44 (93.18%).

DR-grade and research review truth remained explicit:

| Grade | Images | `RETAKE` | `LIMITED` | `ROUTINE_REVIEW` | `PRIORITY_REVIEW` | `UNCERTAIN` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 174 | 9 | 159 | 4 | 0 | 2 |
| 1 | 46 | 0 | 44 | 0 | 0 | 2 |
| 2 | 92 | 4 | 79 | 0 | 1 | 8 |
| 3 | 68 | 7 | 57 | 0 | 4 | 0 |
| 4 | 20 | 3 | 17 | 0 | 0 | 0 |

The full-cohort false-routine danger count was 0/180 grade-derived `PRIORITY`
images; false-priority workload was 0/220 grade-derived `ROUTINE` images.
Blocked and uncertain images remain in those truth-class denominators; they
are not counted as correct decisions. The JSON includes nominal one-sided
exact 95% image bounds (1.65% and 1.35%) only as descriptive summaries: the
correlated dual views do not support an independent-image interpretation.
Patient-event bounds below are the primary uncertainty summaries, though they
also remain exploratory and are not deployment guarantees.

## Patient-level result

Each of the 100 patients has four images. The adverse-event definitions retain
the baselines' patient units:

- technical-quality false `READY`: any true `RETAKE` image released `READY`,
  2/74 RETAKE-bearing patients (2.70%);
- technical-quality false `RETAKE`: any true `READY` image called `RETAKE`,
  1/65 READY-bearing patients (1.54%);
- combined false-routine danger: any grade-derived `PRIORITY` image released
  `ROUTINE_REVIEW`, 0/50 priority-truth patients; one-sided exact upper 95%
  bound 5.82%;
- combined false-priority workload: any grade-derived `ROUTINE` image released
  `PRIORITY_REVIEW`, 0/60 patients containing a routine-truth image; one-sided
  exact upper 95% bound 4.87%.

For a fail-closed patient review queue, `PRIORITY_REVIEW` wins if any image
releases it; `ROUTINE_REVIEW` is released only when all four images release it;
any quality block or downstream abstention otherwise produces `UNCERTAIN`.
That gives 4 `PRIORITY_REVIEW`, 0 `ROUTINE_REVIEW`, and 96 `UNCERTAIN`
patients: 4% patient review coverage.

The JSON also reports a full-state descriptive aggregation that preserves
`RETAKE` versus `LIMITED`: 19 `RETAKE`, 68 `LIMITED`, 0 `ROUTINE_REVIEW`, 4
`PRIORITY_REVIEW`, and 9 `UNCERTAIN`. It is reporting-only and does not change
the application contract.

## Does the quality gate block priority cases?

Yes, under the dataset-derived research truth—not as a clinical claim. The
quality-first contract blocked 167/180 `PRIORITY`-truth images (92.78%): 14 as
`RETAKE` and 153 as `LIMITED`. Only 13/180 reached review-priority evaluation.

The same effect appears without reference to truth: the standalone escalation
baseline would have produced 56 `PRIORITY` decisions across all 400 images,
but the quality gate blocked 51/56 (91.07%), leaving 5 final
`PRIORITY_REVIEW` releases. At patient level, 39/50 priority-truth patients had
every priority-truth image blocked; 11 had at least one such image pass, and 4
received at least one final `PRIORITY_REVIEW` release.

This composition therefore sharply reduces both observed adverse releases and
useful priority coverage. Zero observed false-routine/false-priority events
must be read together with 97.75% non-decision and the wide patient-level
bounds. Composing two historically calibrated selective policies does not
create a new full-pipeline risk guarantee.

## Reproduction

Full deterministic evaluation (hash verification, 400-row join, metrics, and
400 image-level audit records):

```bash
python3 ml/evaluate_combined_pipeline.py
```

Small deterministic smoke (all 400 source rows and hashes are still validated
before evaluating the first eight joined rows):

```bash
python3 ml/evaluate_combined_pipeline.py \
  --limit 8 \
  --output outputs/combined-offline-evaluation/smoke.json
```

Tests:

```bash
python3 -m unittest ml.tests.test_combined_pipeline -v
python3 -m unittest discover -s ml/tests -v
```

Primary machine-readable output:
`outputs/combined-offline-evaluation/report.json`. It includes verified input
hashes, policies, integrity checks, all image- and patient-level denominators,
truth cross-tabs, coverage/abstention, quality-gate effect, limitations, and
all 400 joined records.
