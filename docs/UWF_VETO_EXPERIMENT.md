# Experimental ultra-widefield safety veto

## Decision

**Do not integrate this gate into the live runtime yet.** The experiment shows
compelling separation inside DeepDRiD, but it does not establish robustness to
new conventional-fundus or widefield cameras. The saved artifact is explicitly
marked `runtime_integration_authorized=false` and can only force `LIMITED`; it
can never promote READY or RETAKE.

## Protocol

The frozen-feature ensemble used only DeepDRiD conventional central-field CFP
and UWF data. Every UWF-validation patient was excluded from development,
including their conventional training images. DeepDRiD test and the opened MSHF
test were refused. The threshold was set on a patient-grouped internal
calibration partition using each conventional patient's maximum UWF score.

- Conventional development/evaluation: 1,200 train images and 400 official
  validation images.
- UWF development/evaluation: 154 train images from 77 patients and 50
  official validation images from 25 patients.
- Development/evaluation patient overlap: zero.
- Policy: score strictly above 0.614858 forces LIMITED; otherwise the existing
  quality gate is unchanged.

## Results

On the official DeepDRiD validation cohorts, ranking AUC was 1.000. The veto
caught 50/50 UWF images and 25/25 UWF patients. It falsely vetoed 4/400 (1.0%)
conventional images and 3/100 (3.0%) conventional patients. Exact 95% bounds
were a 94.18% lower bound on image-level UWF recall and a 2.27% upper bound on
the conventional image false-veto rate.

A second paired holdout used the 25 UWF-validation patients' 100 conventional
training images, all excluded from model development. It caught 50/50 UWF
images, falsely vetoed 3/100 conventional images, and achieved clean separation
for 23/25 patients.

## Why it remains experimental

- Only 25 UWF validation patients are available.
- Perfect ranking after one tuning epoch strongly suggests easy camera, border,
  color, or cohort shortcuts may contribute.
- The paired holdout comes from the same DeepDRiD acquisition ecosystem.
- Internal calibration patients are not historically fresh, so its nominal
  finite-sample bound is not a deployment guarantee.
- A true runtime gate needs a separately sourced, multi-device modality cohort
  and an in-scope regression showing it does not veto acceptable conventional
  cameras.

Reproduce with `ml/experiment_uwf_veto_gate.py`. Full results and artifact flags
are in `outputs/uwf-veto-gate/report.json` and
`outputs/uwf-veto-gate/uwf-veto-experiment.pt`.
