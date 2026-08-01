# Evaluation ledger

This ledger prevents results from different runtimes and split histories from
being presented as though they were one clinical evaluation.

| Split / source | Current status | Permitted use |
| --- | --- | --- |
| DeepDRiD train, 300 patients | Used across earlier specialist/Gemma fitting and threshold experiments; no patient remains historically fresh | Training and post-hoc internal evidence only |
| DeepDRiD validation, 100 patients | Viewed repeatedly during development; complete hybrid run recorded | Exploratory comparison, end-to-end reliability check, and demo examples only |
| DeepDRiD test, 100 patients | Opened once for the frozen Gemma smoke adapter; not used by the specialist | Historical Gemma result only; do not reuse for iteration |
| MSHF author-provided test, 260 images | Evaluated once with the frozen specialist; no MSHF fitting or threshold tuning | External image-level stress result; now opened, so future iteration must not call it untouched |
| Escalation quality-pass val, 70 images / 25 patients | Used only to rank the five audited Gemma escalation candidates | Frozen candidate selection; do not relabel as test evidence |
| Escalation quality-pass calibration, 212 images / 70 patients | Opened after adapter selection to freeze the three-way policy | Threshold calibration and stated research-risk calculation only |
| Escalation quality-pass eval, 182 images / 65 patients | Opened after adapter and thresholds were frozen | Held-out direct-logit research evaluation; now opened, never reuse for iteration |

The paper-inspired spatial/color challenger used only the internal DeepDRiD
train/validation development protocol. The global baseline retained AUC 0.803
and 11.0% selective coverage. Global-plus-2x2-spatial fell to AUC 0.792 and
8.75% coverage; adding color statistics fell to AUC 0.791 and 8.25% coverage.
Neither was promoted. See `outputs/spatial-color-challenger/report.json`.

The veto-only UWF experiment used DeepDRiD regular-CFP and UWF train/validation
data with zero development/evaluation patient overlap, and refused DeepDRiD
test and MSHF. It caught 50/50 UWF validation images while falsely vetoing
4/400 conventional validation images. Because the UWF cohort has only 25
patients and may expose acquisition shortcuts, the artifact is experimental
and not integrated. See `docs/UWF_VETO_EXPERIMENT.md`.

## Frozen MSHF external stress result

The frozen specialist was evaluated once on MSHF's author-provided 260-image
test directory. Labels were reconstructed by majority vote from all three
released annotators. MSHF exposes no patient identifier, so these are
image-level metrics and provide no patient-level risk guarantee.

| Camera group | Images | ROC-AUC (decision score) | Selective coverage | Accepted accuracy | False READY |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional CFP | 100 | 0.784 | 12.0% | 12/12 (100%) | 0/50 |
| Portable camera | 60 | 0.681 | 0% | n/a | 0/48 |
| Ultrawide-field mosaic | 100 | 0.470 | 11.0% | 7/11 (63.6%) | 4/42 |
| **All MSHF** | **260** | **0.700** | **8.85%** | **19/23 (82.6%)** | **4/140 (2.86%)** |

The small conventional-CFP group is encouraging, but it is not validation of
the intended scope. The result also reveals a concrete failure mode: the
current model does not generalize to UWF mosaics. UWF must be treated as out of
scope until a separately trained and validated modality gate exists; the live
runtime does not yet detect it automatically. The model abstained on all 60
portable-camera inputs, so this test provides no safety or utility estimate for
that device class. The complete machine-readable report is
`outputs/mshf-external-test-specialist.json`.

## Full local hybrid validation run

The actual loopback application pipeline was run over all 400 DeepDRiD
validation images: upload API, frozen specialist, selective gate, local Gemma 4
26B confirmation/veto, schema normalization, and final application policy. All
400 images remain in the denominator, including abstentions and any runtime
failure. This split was viewed during development, so the result is an
exploratory end-to-end reliability check rather than a fresh performance test.

- Decisions: 20 READY, 23 RETAKE, and 357 LIMITED; 10.75% coverage.
- Accepted accuracy: 40/43 (93.02%).
- Image-level false READY: 2/218 (0.92%); false RETAKE: 1/182 (0.55%).
- Patient-event false READY: 2/74 (2.70%); false RETAKE: 1/65 (1.54%).
- Flow: 356 specialist abstentions, 43 Gemma-confirmed decisions, one Gemma
  veto/internal abstention, and zero HTTP or schema failures.
- End-to-end API latency: 168 ms median and 5.91 s p95 over all 400 images on
  the current 24 GB Mac. This is dominated by 356 specialist short-circuits;
  among the 44 Gemma-invoked cases, median was 5.77 s and p95 was 9.26 s.
- Quality attention: present on all 23 final RETAKE decisions, absent on every
  READY and LIMITED decision, with zero generation/schema failures.
- Decision trace: valid on 400/400 responses; 356 abstained/skipped, 20 READY
  confirmed, 23 RETAKE confirmed, and one READY candidate without confirmation
  normalized to LIMITED.

These numbers are almost the specialist policy result because Gemma may veto
or abstain but never promote a specialist `LIMITED` decision. They demonstrate
that the complete local application preserved the safety behavior without
silently dropping failures. The machine-readable report, including all 400
records and the verified runtime identity, is
`outputs/hybrid-validation-exploratory.json`.

## Frozen RetinaPriority escalation result

The review-priority task uses separately derived quality-pass-only partitions.
Candidate choice used the complete 70-image validation set only. The selected
q/v rank-16 step-60 adapter scored priority-positive AUC 0.92375; the q/v
challenger scored 0.895, and the larger q/k/v/o root, checkpoint 74, and
checkpoint 111 scored 0.880, 0.868333, and 0.880.

After selection, the complete 212-image calibration partition froze a strict
three-way policy. The 182-image, 65-patient evaluation partition then produced:

- direct-logit priority-positive ROC-AUC 0.956903;
- balanced accuracy 0.912355 at threshold 0.5;
- 36 Priority, 29 Routine, and 117 Uncertain under the selective policy;
- 35.7% image coverage and 65/65 accepted routes correct;
- 36.9% patient coverage and 24/24 accepted aggregate routes correct.

The calibration supports a stated 10% patient-event risk target per error type
with 90% simultaneous confidence as research evidence. It cannot certify 5%.
The live llama.cpp free-generation demo is a different inference contract and
must not inherit the offline thresholds, coverage, or confidence statement.
Exact reports and hashes are recorded in
`models/retinapriority-gemma4-26b/manifest.json`.

## Non-comparable result families

- **HF decision logits:** deterministic READY-vs-RETAKE class-token ranking
  from the Gemma PEFT checkpoint.
- **llama.cpp free generation:** structured JSON produced by local Gemma; prompt,
  decoding, quantization, and schema normalization affect the result.
- **Quality specialist:** frozen DenseNet features plus the multi-task head and
  an abstaining policy.
- **Hybrid:** specialist gate plus Gemma confirmation/veto. Its full 400-image
  validation run is recorded separately from specialist-only metrics.
- **RetinaPriority direct logits:** deterministic ROUTINE-vs-PRIORITY class-token
  ranking from the selected Gemma PEFT checkpoint after the separate quality
  gate.
- **RetinaPriority live free generation:** schema-constrained labels from the
  converted llama.cpp LoRA for pinned demo inputs; explicitly uncalibrated.

Do not merge their metrics. In the earlier quality-confirmation hybrid, Gemma
could only veto, so it could not promote a `LIMITED` specialist decision. In
the current combined product, RetinaPriority runs only after exact `READY`, and
its ROUTINE/PRIORITY evidence remains separate from the quality-gate metrics.

## Terminology

- `false READY rate` is false READY divided by all truly RETAKE examples (or
  RETAKE-bearing patients for patient-event metrics).
- `READY precision` is true READY divided by all READY calls. It answers a
  different question.
- `decision score` is an uncalibrated ranking score, not a probability.
- `coverage` is the fraction receiving READY or RETAKE rather than LIMITED.
- Exact bounds require a frozen pipeline and a genuinely fresh, exchangeable
  sample at the stated calibration unit. The current DeepDRiD patient bounds
  are nominal/post-hoc because those patients influenced earlier experiments;
  they are not deployment, regulatory, or clinical guarantees.
