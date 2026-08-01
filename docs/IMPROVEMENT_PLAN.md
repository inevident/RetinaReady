# RetinaReady accuracy and winning-demo plan

## Recommendation

Use a **hybrid, selective quality gate**:

1. A small retinal-quality specialist reads low-level image signals and outputs
   an overall READY score plus artifact, clarity, and field-definition scores.
2. A conservative patient-level research policy converts the overall score
   into READY, RETAKE, or LIMITED after the model is frozen. Its current exact
   bounds are post-hoc evidence, not a fresh risk guarantee.
3. Gemma 4 receives the image and the specialist evidence, independently
   verifies every candidate decision, and can veto a conflicting or unsupported
   result. On a confirmed RETAKE it may add normalized technical issue codes.
4. Application code owns the final safety policy and deterministic recapture
   instructions. Free-form model text never promotes an uncertain image.

This keeps Gemma central to the interaction while giving low-level retinal
perception to the component that performs it best. It also makes the product
claim stronger: in the default loopback configuration the system is local,
fast, provides factor-level rationale, and is designed to abstain—not merely a
binary VLM prompt.

## Why the first adapter plateaued

The completed adapter is a valid integration smoke, not a final training run:

- It used only 128 of 1,200 training images.
- Sixty optimizer steps equal 3.75 passes over that small subset.
- Validation loss was best at step 20 and deteriorated afterward, consistent
  with overfitting and increasingly saturated scores.
- Its loss supervises only one class token. DeepDRiD's artifact, clarity, and
  field-definition annotations receive no loss.
- PEFT adapted language q/v projections while the vision encoder and projector
  stayed frozen.
- The bounded subset was label-balanced but not balanced by patient, image
  view, or study source.

At the ordinary 0.5 threshold, the model has useful but moderate ranking
ability. The low READY recall at the old 0.988 threshold is therefore mainly a
policy tradeoff, not a thresholding bug: calibration cannot change the ROC
ranking.

| Model / policy | Split | ROC-AUC | Balanced accuracy | READY recall | False READY |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma smoke adapter, threshold 0.5 | Validation, 400 | 0.750 | 68.9% | 73.1% | 35.3% |
| Gemma smoke adapter, threshold 0.988 | Test, 400 (already opened) | 0.756 | 61.1% | 29.4% | 7.3% |
| Published QuickQual-MEME head, threshold 0.5 | Validation, 400 | 0.765 | 69.4% | 74.2% | 35.3% |
| Frozen multi-task specialist, threshold 0.5 | Validation, 400 (exploratory) | **0.803** | **74.2%** | **76.4%** | 28.0% |

Metrics across papers and datasets are not directly comparable. False READY
means the fraction of true RETAKE images incorrectly released as READY; it is
not the fraction of READY calls that are wrong.

## What recent work says to copy

### Multi-task acquisition factors

The 2026 paper
[Semi-supervised multi-task learning for interpretable quality assessment of fundus images](https://www.sciencedirect.com/science/article/pii/S1746809425016787)
reports DeepDRiD F1 improving from 0.763 to 0.778 when a ResNet-18 learns
illumination, clarity, and contrast alongside overall quality. The detail
outputs also support actionable recapture feedback. This is closely aligned
with our contract, though its metrics are not directly comparable to ours.

A 2024 multi-center
[refined CFP quality assessment study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10903193/)
uses location, clarity, artifact, optic-disc/macula location, laterality, and
field of view in one network. It reports 88.15% internal and 86.63% external
overall accuracy on its own much larger dataset. Its important lesson is the
structured objective, not direct comparability of its headline number.

### Global and local image evidence

The 2026
[SAM-IQA retinal study](https://www.sciencedirect.com/science/article/abs/pii/S0014483526000941)
combines global-image and salient local-patch branches and reports 81.5%
DeepDRiD AUC. This supports adding one global fundus view plus a small number of
quality-sensitive local views instead of asking Gemma to infer every defect
from one resized image.

### Continuous scores and spatial feedback

The 2025
[FTHNet/FQS work](https://www.nature.com/articles/s41598-025-24423-8)
releases 2,246 images with continuous 0–100 expert opinion scores and trains a
small regression model. For RetinaReady, its continuous target suggests a
future risk/workload slider; that product use is our inference, not a result
established by the paper.

The 2026
[EFIQA paper](https://proceedings.mlr.press/v315/wang26f.html)
uses masked anatomical inpainting and a shallow adapter to create pixel-level
quality maps without quality-label supervision. We recommend testing a
localized “quality degraded here” overlay next; the paper does not establish
that this will improve RetinaReady's cross-dataset results.

### DeepDRiD is genuinely difficult

The official [DeepDRiD challenge report](https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/)
describes 2,000 regular fundus images from 500 patients and reports quality
accuracy of roughly 65–70% among challenge submissions. Our 74.2% balanced
validation accuracy uses a different metric and development protocol, so it is
an encouraging internal result—not evidence that we beat those submissions.
Careful patient grouping and external validation remain essential.

## Implemented improvement

`ml/train_quality_specialist.py` now:

- extracts frozen ImageNet DenseNet-121 features using the public QuickQual
  crop/pad/resize normalization;
- splits DeepDRiD training data into patient-disjoint fit, tuning, and
  within-run calibration partitions, stratified by each patient's READY-image
  pattern;
- selects stopping epochs on tuning patients, then refits the multi-task
  ensemble on fit+tuning before touching calibration;
- sets separate READY and RETAKE thresholds with patient-level one-sided exact
  binomial calculations and a nominal joint 95% error budget;
- refuses the already-open `test.csv` manifest;
- saves a 1.3 MB head artifact and a complete reproducibility report.

The deployed bundle uses one shared multi-task ensemble for the decision and
factor roles. The 31 MB DenseNet backbone and two local copies of the head
artifact total about 33 MB. A recorded three-image CPU acceptance check ran in
123–153 ms per image after loading; this is a machine smoke, not a latency
distribution.

On the official 400-image validation split, which was not used to fit or
calibrate the head but was viewed during earlier architecture work, these
results are exploratory:

- ROC-AUC: **0.8028**
- Accuracy / balanced accuracy at 0.5: **74.0% / 74.20%**
- READY recall at 0.5: **76.37%**
- False READY at 0.5: **27.98%**

The frozen three-way policy produced on that exploratory validation split:

- 21 READY, 23 RETAKE, 356 LIMITED
- **11.0% coverage**
- **93.18% accuracy** among READY/RETAKE decisions
- **0.92% false READY** among true RETAKE images
- **0.55% false RETAKE** among true READY images
- Patient-event rates of 2.70% false READY and 1.54% false RETAKE

The multi-task factor head's exploratory validation mean absolute errors on a
0–100 scale were approximately 12.0 for artifact, 10.4 for clarity, and 8.7 for
field definition. These are technical acquisition estimates, not clinical
findings. The `<65` issue tags are an unevaluated UI heuristic.

The threshold-setting set contained 74 RETAKE-bearing and 70 READY-bearing
patients. Each gate observed two patient errors; its nominal one-sided exact
upper bound was 9.42% and 9.94%, respectively, with delta 0.025 per gate. The
mathematics would yield at least 95% simultaneous confidence for a frozen
pipeline and genuinely fresh exchangeable patients. That condition is not met
here: all DeepDRiD training patients influenced earlier project experiments.
The current values are therefore post-hoc evidence, not a risk guarantee. They
also concern an any-error event over the fixed four-image DeepDRiD protocol,
not arbitrary numbers or mixes of views. The raw sigmoid is not a calibrated
probability.

## Live hybrid behavior

`app/quality_specialist.py` and `HybridLocalAnalyzer` now run the complete
pipeline offline:

```text
fundus image
  -> 33 MB retinal-quality specialist
  -> frozen conservative READY / RETAKE / LIMITED gate
  -> image + factor evidence sent to local Gemma 4
  -> Gemma independent confirmation or safety veto
  -> deterministic browser response
```

The local hybrid first verified READY, RETAKE, and LIMITED behavior on three
curated DeepDRiD examples. It was then exercised through the real upload API on
all 400 validation images. The run made 20 READY, 23 RETAKE, and 357 LIMITED
decisions: **10.75% coverage** and **93.02% accepted accuracy**. It produced
2/218 false READY and 1/182 false RETAKE image decisions. Gemma confirmed 43
specialist decisions, vetoed or internally abstained once, and the run had zero
HTTP/schema failures. Median end-to-end latency was 168 ms because most images
short-circuited at LIMITED; p95 was 5.91 seconds because accepted cases invoked
the 26B model. The 44 Gemma-invoked cases had 5.77-second median and 9.26-second
p95 latency. A Gemma timeout or disagreement returns LIMITED rather than an
HTTP failure or a browser-invented result.

The validation split was viewed during development, so these are exploratory
end-to-end reliability metrics, not a fresh clinical estimate or guarantee.
The curated smoke is in `outputs/hybrid-runtime-acceptance-20260731.json`; all
400 records and the verified runtime identity are in
`outputs/hybrid-validation-exploratory.json`.

For RETAKE decisions, the app now computes a factor-specific quality-attention
overlay after the decision pass. It targets the weakest acquisition factor,
falls back from signed Grad-CAM to gradient sensitivity if needed, and is
omitted on failure. READY and LIMITED skip it. The overlay is explanatory only,
explicitly labeled as non-pathology localization, and cannot alter or promote a
decision. A real-model smoke placed its extra pass at roughly 0.4 seconds; the
refreshed 400-image runtime run included it. All 23 final RETAKE decisions had
an attention map and no READY or LIMITED decision did. All 400 responses also
passed the enumerated, fail-closed decision-trace contract.

## Completed paper-inspired spatial/color challenger

A reversible frozen-feature challenger tested a 2x2 local feature grid inspired
by global/local RIQA work, with and without fixed color statistics. It used the
same patient-grouped internal fit/tune/calibration protocol and did not touch
DeepDRiD test or the opened MSHF test. Neither variant earned deployment:

- global baseline: AUC 0.803, 74.20% balanced accuracy, 11.0% selective
  coverage, and 93.18% accepted accuracy;
- global plus 2x2 spatial: AUC 0.792, 73.19% balanced accuracy, 8.75% coverage,
  and 97.14% accepted accuracy;
- spatial plus color statistics: AUC 0.791, 72.14% balanced accuracy, 8.25%
  coverage, and 90.91% accepted accuracy.

Both challengers slightly improved mean factor MAE but reduced ranking and
coverage. The deployed global baseline therefore remains frozen. This negative
result is useful evidence that the project applies a promotion rule instead of
selecting whichever experiment sounds most novel. The complete report is
`outputs/spatial-color-challenger/report.json`.

## One-time frozen external device-shift stress test

The frozen specialist was evaluated once, without MSHF fitting or threshold
changes, on the authors' 260-image test directory. Overall ROC-AUC was 0.700;
the conservative policy covered 23/260 images and got 19/23 accepted decisions
correct, with 4/140 false READY and 0/120 false RETAKE.

The aggregate hides the useful finding:

- conventional CFP: AUC 0.784 and 12/12 accepted decisions correct on this
  small 100-image group, which is encouraging but not validation;
- portable cameras: AUC 0.681 but 0% automated coverage, so safety and utility
  on that device class remain unknown;
- UWF mosaics: AUC 0.470 and all four external false READY calls.

MSHF does not expose patient identifiers, so these are image-level stress
metrics, not patient-level guarantees. The test is now opened and must not be
used for iterative tuning. The product scope should remain conventional
central-field color fundus photography; UWF requires a separately developed
modality/OOD gate; the current runtime does not yet detect UWF automatically.
Full results are in
`outputs/mshf-external-test-specialist.json`.

## Experimental UWF safety veto

A patient-grouped, veto-only modality experiment used DeepDRiD's regular-CFP
and UWF training/validation sources without touching DeepDRiD test or MSHF. On
the official validation cohorts it detected 50/50 UWF images and falsely vetoed
4/400 conventional images. A paired held-out cohort from the same 25 patients
detected 50/50 UWF images and falsely vetoed 3/100 conventional images.

Despite AUC 1.000, it is not integrated. The small cohort and near-trivial
separation may reflect camera, border, or color shortcuts. The artifact is
veto-only and explicitly marked as unauthorized for runtime integration. A
multi-device, separately sourced modality cohort is required before promotion.
See `docs/UWF_VETO_EXPERIMENT.md`.

## Next A100 experiment

`ml/configs/gemma4_26b_decision_full.json` is ready and passes its local dry
run. Compared with the smoke adapter it will:

- train on all 1,200 images and evaluate on all 400 validation images;
- run three epochs rather than 60 steps over a 128-image subset;
- lower learning rate from 2e-4 to 5e-5 with cosine decay;
- adapt language q/k/v/o projections rather than q/v only;
- use LoRA alpha 32 and load the best validation-loss checkpoint at the end;
- preserve pinned model, processor, dataset, and provenance checks.

Extrapolating from the prior 60-step run suggests roughly 73 minutes for 450
steps before evaluation/conversion overhead; budget 50–90 minutes on an A100
80 GB. Do not touch the official test
again while choosing architecture, learning rate, or thresholds.

After this decision run, the higher-ceiling experiment is a multi-task Gemma
target with equal single-token internal class codes and supervised quality
factors. Vision-tower LoRA is scientifically attractive but should wait until
its llama.cpp conversion and local projector compatibility are proven.

## Remaining high-value work

1. If global/local evidence is revisited, use a lower-capacity learned
   attention or residual-spatial projection; the completed fixed 2x2/color
   variants did not pass promotion.
2. Add controlled train-only blur, exposure, glare, contrast, and field-cutoff
   augmentation; preserve the unmodified samples and avoid teaching pathology
   as an acquisition defect.
3. Compare the implemented Grad-CAM-style quality overlay with a true
   EFIQA-style map later; keep both explanation-only and never use them to
   promote a decision.
4. Validate the completed UWF veto challenger on a separately sourced,
   multi-device modality cohort; do not tune it on the now-opened MSHF test.
   Until then, keep it outside the runtime and state the supported scope.
5. After any change, freeze one pipeline, calibrate once on fresh patients, and
   report one-sided confidence bounds, coverage, selective accuracy, and
   low-FPR partial AUC.

## Judge-facing story

> RetinaReady does not diagnose retinal disease. It helps the operator capture
> a technically reviewable image, explains the acquisition defect, and knows
> when the evidence is insufficient. In the default configuration, the
> Gemma-powered workflow runs over loopback and the application does not
> intentionally persist or upload the patient image.

Demo three examples: a clear READY image, an obvious RETAKE with two concrete
fixes, and an ambiguous LIMITED case. Then move a precomputed risk/workload
slider to show that abstention is a deliberate safety feature rather than a
failure.

Exact artifact hashes, split counts, calibration assumptions, and limitations
are recorded in `docs/QUALITY_SPECIALIST_MODEL_CARD.md`; split history and
non-comparable metric families are recorded in `docs/EVALUATION_LEDGER.md`.
