# RetinaReady quality specialist model card

## Intended use

This 33 MB offline component ranks the **technical capture quality** of color
fundus photographs. It produces a decision score and three acquisition-factor
scores: artifact quality, clarity, and field definition. A selective policy
maps the decision score to `READY`, `RETAKE`, or `LIMITED`.

It is a hackathon research prototype. It does not diagnose disease, determine
that an eye is healthy, or replace clinician review. The factor-to-issue rule
(`< 65` becomes a UI issue tag) is an unevaluated recapture heuristic, not a
DeepDRiD label or clinical threshold.

## Frozen artifacts

| File | SHA-256 | Role |
| --- | --- | --- |
| `densenet121-a639ec97.pth` | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` | Torchvision DenseNet-121 ImageNet-1K V1 backbone |
| `decision-head.pt` | `84081ad06122a0354d0bd4c31cdc53052f1bdb4999fb706b7babcfe72b94d936` | Five-member multi-task MLP ensemble and policy |
| `factor-head.pt` | `84081ad06122a0354d0bd4c31cdc53052f1bdb4999fb706b7babcfe72b94d936` | Same frozen multi-task ensemble, loaded for factor outputs |

The two head files are intentionally identical in this bundle: recent
multi-task retinal-IQA evidence motivated one shared representation, and the
same frozen ensemble performed both roles. Review the upstream torchvision
and ImageNet weight terms before redistributing the backbone.

## Data and selection protocol

DeepDRiD training patients were stratified by their number of READY images and
then split by patient—not image:

| Partition | Patients | Images | Use |
| --- | ---: | ---: | --- |
| Fit | 158 | 632 | Initial head training |
| Tuning | 35 | 140 | Select stopping epoch for each ensemble member |
| Final development | 193 | 772 | Refit from scratch for the frozen epochs |
| Calibration | 107 | 428 | Threshold setting after model freeze in this run; not historically fresh |
| Official validation | 100 | 400 | Exploratory evaluation only |

No calibration patient enters fit, tuning, or final development **within this
run**. However, all 107 patients influenced the broader project history: 60
were in the older calibration split and the other 47 participated in earlier
fitting experiments. The official test manifest is rejected by the specialist
trainer and was not used for this specialist. The official validation split
was viewed during earlier architecture work, so its numbers below are
model-development evidence, not a final untouched test.

## Patient-level selective policy (nominal post-hoc evidence)

The READY event is defined as **any** truly RETAKE image from one patient being
released as READY. The RETAKE event is the dual: any truly READY image from one
patient being sent to RETAKE. Each patient therefore contributes at most one
Bernoulli event per gate.

- READY threshold: decision score strictly greater than `0.9498774409294128`.
- RETAKE threshold: decision score strictly less than `0.015006430447101593`.
- Everything between the thresholds is `LIMITED`.
- Target upper risk: 10% for each gate.
- One-sided delta: 0.025 per gate. If these had been genuinely fresh,
  exchangeable calibration patients, the union bound would give at least 95%
  simultaneous confidence for both bounds.
- Calibration units: 74 RETAKE-bearing and 70 READY-bearing patients.
- Calibration errors: 2 false-READY patient events and 2 false-RETAKE patient
  events.
- Exact upper bounds: 9.42% and 9.94%, respectively.

The calculations are mechanically correct, but historical patient reuse makes
the 9.42% and 9.94% values **nominal post-hoc bounds, not a finite-sample
guarantee**. The event also assumes DeepDRiD's fixed four-image patient bundle
and the same outcome-stratum mix; changing views, image count, device, clinic,
or prevalence changes the estimand. A defensible guarantee requires freezing
the entire pipeline and calibrating once on a new, prospectively sampled
patient cohort independent of outcomes. These results are not clinical
validation and the raw sigmoid is not a calibrated probability. The API
therefore reports `confidence: null`.

## Exploratory validation results

At the ordinary 0.5 decision threshold:

- ROC-AUC: 0.8028.
- Accuracy: 74.0%; balanced accuracy: 74.20%.
- READY recall: 76.37%; RETAKE recall: 72.02%.
- False-READY rate among RETAKE images: 27.98%.

With the frozen three-way selective policy:

- 21 READY, 23 RETAKE, and 356 LIMITED decisions; 11.0% coverage.
- 93.18% accuracy among non-abstained images.
- Image-level false READY: 2/218 (0.92%); false RETAKE: 1/182 (0.55%).
- Patient event false READY: 2/74 (2.70%); false RETAKE: 1/65 (1.54%).

Factor mean absolute errors on the 0–100 scale were 11.99 for artifact
quality, 10.39 for clarity, and 8.74 for field definition.

For RETAKE decisions, the live app can render a Grad-CAM-style attention
overlay for the weakest predicted factor. READY and LIMITED skip this pass so
the UI does not imply a defect or over-explain uncertainty. It is computed only
after the frozen gate, is omitted on any error, and never changes a decision.
The label explicitly says “Model quality attention — not pathology
localization.” This is an unvalidated explanatory visualization, not evidence
of disease or lesion boundaries.

A presentation audit found that all 23 final RETAKE maps targeted artifact as
the weakest predicted factor, while artifact was lowest or tied-lowest in the
released DeepDRiD factor labels for 6/23 cases. The live UI therefore says
“weakest predicted factor,” and the demo must not claim causal defect
localization. See `docs/RETAKE_EXPLANATION_AUDIT.md`.

## Runtime and provenance

The backbone and heads load in about one second on the current 24 GB Mac. A
three-image CPU acceptance check measured 123–153 ms per image after load and
reproduced READY, LIMITED, and RETAKE examples. The machine-readable check is
`outputs/specialist-runtime-acceptance-20260731.json`.

Canonical training reports:

- `outputs/quality-specialist-rigorous-factors/report.json`
- `outputs/quality-specialist-rigorous-decision/report.json`
- `ml/train_quality_specialist.py`
- `ml/calibrate_selective_policy.py`

## End-to-end hybrid runtime check

The frozen specialist was also evaluated inside the complete loopback
application with local Gemma 4 26B on all 400 DeepDRiD validation images. The
hybrid returned 20 READY, 23 RETAKE, and 357 LIMITED decisions: 10.75% coverage
and 40/43 (93.02%) accepted accuracy. There were 2/218 false READY and 1/182
false RETAKE image decisions. Gemma confirmed 43 decisions and vetoed or
internally abstained once; no upload, HTTP, or schema failure occurred. Median
end-to-end API latency was 168 ms and p95 was 5.91 seconds on the current 24 GB
Mac, dominated by 356 fast specialist abstentions. Among the 44 Gemma-invoked
cases, median latency was 5.77 seconds and p95 was 9.26 seconds. All 23 final
RETAKE responses contained the optional factor-specific quality-attention map;
no READY or LIMITED response did. The map is explanation-only and cannot alter
the gate. The deterministic trace was valid on all 400 responses: 356
specialist-abstained/Gemma-skipped paths, 43 confirmed candidates, and one
unconfirmed READY candidate safely normalized to LIMITED.

All abstentions and failures remain in the denominator. This is still an
exploratory result because DeepDRiD validation was already viewed during
development. It does not repair the historical calibration reuse described
above. See `outputs/hybrid-validation-exploratory.json` and
`ml/evaluate_hybrid_runtime.py`.

## External MSHF stress test

Without fitting or changing thresholds on MSHF, the frozen specialist was run
once on the dataset authors' 260-image test directory. Overall ROC-AUC was
0.700. The selective policy made 23 decisions (8.85% coverage), with 19/23
correct, 4/140 false READY, and 0/120 false RETAKE. Results differed sharply by
camera group:

- Conventional CFP: AUC 0.784; 12/12 accepted decisions correct.
- Portable camera: AUC 0.681; all 60 images abstained.
- Ultrawide-field mosaic: AUC 0.470; 7/11 accepted decisions correct, including
  4 false READY calls.

These are image-level stress metrics because the release does not expose
patient identifiers. They do not recalibrate or validate the DeepDRiD policy.
The UWF result supports treating that modality as out of scope; it is not a
reason to tune on the opened MSHF test set. The runtime does not yet detect UWF
automatically. See `outputs/mshf-external-test-specialist.json` and
`ml/evaluate_external_mshf.py`.

## Known gaps

The completed MSHF stress test suggests useful external conventional-CFP
ranking on a small group, leaves portable-camera usefulness unknown, and
supports treating UWF as out of scope.
Planned work includes an independently trained modality/OOD gate,
global-plus-local views, controlled capture-defect augmentation, spatial
quality maps, reliability calibration, and prospective operator testing. A
fresh prospective patient cohort is required before any risk guarantee. Any
change to the model, thresholds, preprocessing, view protocol, or target
population also requires new calibration.
