# Gemma escalation calibration and evaluation

`ml/calibrate_escalation_adapter.py` turns two complete Gemma escalation
decision-logit reports into a frozen, three-way review-priority policy:

- score strictly below the lower threshold: `ROUTINE`
- score strictly above the upper threshold: `PRIORITY`
- either threshold exactly, or anything between them: `UNCERTAIN`

`ROUTINE` is only a lower place in a clinician review queue. It is not a
finding that the retina is healthy. `PRIORITY` is not a diagnosis, validated
urgency determination, or treatment recommendation.

## Required sequence

First, run `ml/evaluate_decision_logits.py --task escalation` over **every**
row of the calibration manifest. Run it again over **every** row of the
patient-disjoint evaluation manifest. Do not pass `--limit` to either run.
Then freeze and evaluate the policy:

```bash
python3 ml/calibrate_escalation_adapter.py \
  --calibration-report outputs/a100-retinapriority-20260801/selected-smoke-adapter/calibration-quality-pass-decision-logits.json \
  --evaluation-report outputs/a100-retinapriority-20260801/selected-smoke-adapter/eval-quality-pass-decision-logits.json \
  --calibration-manifest data/escalation-quality-pass-manifests/calibration.csv \
  --evaluation-manifest data/escalation-quality-pass-manifests/eval.csv \
  --output /tmp/retinapriority-selective-policy-report.json
```

The frozen local reference report is
`outputs/a100-retinapriority-20260801/selected-smoke-adapter/selective-policy-quality-pass-report.json`.
Use a scratch output, as above, when reproducing it so the frozen artifact is
not overwritten.

Thresholds are fitted once from calibration rows. Evaluation rows are loaded
only after their provenance is checked and are never passed to threshold
selection. The utility rejects partial reports, duplicate or missing image
IDs, truth/patient/path mismatches, manifest hash mismatches, an adapter
binding mismatch, the wrong task, and any patient or image overlap between
calibration and evaluation.

The report binds both source reports, both manifests, the adapter weights, the
training provenance, model and processor revisions, and its own canonical
content with SHA-256. Older decision-logit reports that predate the explicit
`expected_split` field remain readable only when the complete report can be
matched to a hash-verified manifest whose every row carries the required
split.

The exported `review_priority_score` is the Gemma two-token softmax output
renamed deliberately: it is an uncalibrated ranking score, not a probability
of disease or harm. The resulting artifact remains research-demo evidence and
does not independently authorize runtime or clinical use.

## Frozen RetinaPriority result

For the validation-selected q/v step-60 adapter, the complete 212-image
calibration report froze these strict thresholds:

- `ROUTINE` when score `< 0.0002611903190957194`
- `PRIORITY` when score `> 0.9993736658418905`
- `UNCERTAIN` otherwise, including equality

The patient-disjoint evaluation contained 182 images from 65 patients. Ranking
AUC was 0.956903. The selective policy accepted 65 images (35.7% coverage), all
65 correctly: 36 Priority and 29 Routine. At patient aggregation it accepted
24/65 patients and all 24 routes were correct.

The configured research target is 10% patient-event risk for each error type
with per-gate delta 0.05, yielding 90% simultaneous confidence. Zero calibration
events gave exact one-sided upper bounds of 7.22% for false Priority and 8.20%
for false Routine. There were not enough calibration patients to certify 5%.
These figures remain research evidence and are not a prospective clinical
guarantee.
