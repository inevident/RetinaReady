# RetinaReady

RetinaReady + RetinaPriority is a fully local, quality-first retinal workflow.
A compact specialist first checks the **technical quality** of a conventional
color fundus photograph while the patient is still present. Only an exact
`READY` result reaches a separate Gemma 4 26B LoRA that suggests
`ROUTINE_REVIEW` or `PRIORITY_REVIEW`; `LIMITED` and `RETAKE` block that stage.

It does **not** diagnose eye disease. `READY` means technically reviewable, not
healthy, and Priority means faster clinician review—not a disease label or a
treatment recommendation.

## What is already built

- A responsive combined interface with four pinned dataset cases that visibly
  demonstrate Routine, Priority, Limited, and Retake paths.
- An in-memory FastAPI endpoint; uploaded images are not written to disk.
- The official DeepDRiD v1.1 release, checksum-verified and converted into
  patient-disjoint train, validation, and test manifests. The test split was
  opened once for the earlier frozen Gemma evaluation and is no longer used for
  iteration.
- Google's official Gemma 4 26B-A4B QAT Q4_0 GGUF and multimodal projector.
- Fully local Gemma inference through `llama.cpp`, bound to `127.0.0.1`.
- A completed Gemma 4 26B decision-token LoRA run on an A100 80 GB, including
  the adapter, provenance, logs, validation/test reports, and a smaller E2B
  fallback configuration.
- A tuned offline llama.cpp bundle: exact-base Q4_0 and Q4_K_M GGUFs, their
  matching BF16 vision projector, and the trained adapter preserved as a
  separate F32 LoRA GGUF so quantization does not erase its learned updates.
- A passed local acceptance gate for the tuned Q4_0/GPU-projector/LoRA profile
  on this 24-GB Mac. Q4_K_M is retained for compatibility experiments but is
  not supported for full-Metal vision inference on this machine.
- A 33-MB retinal-quality specialist trained with patient-disjoint
  fit/tune/calibrate partitions within the current run. Its multi-task head
  provides artifact, clarity, and field scores; a conservative post-hoc policy
  abstains on most borderline images. Because every DeepDRiD training patient
  influenced earlier experiments, its nominal patient-level bounds are not a
  fresh 95% guarantee. It is now the first-stage gate; it does not ask the 26B
  model to prioritize an image that should be recaptured.
- A separate RetinaPriority Gemma 4 q/v rank-16 LoRA, selected only on a frozen
  70-image validation manifest. On a separate 182-image patient-disjoint
  quality-pass evaluation, direct decision logits achieved ROC-AUC 0.956903
  and balanced accuracy 0.912355 at threshold 0.5.
- A separately calibrated three-way priority policy. It accepted 65/182 held-
  out images (35.7% coverage)—36 Priority and 29 Routine—and all 65 accepted
  routes were correct. This is direct-logit research evidence, not a clinical
  validation or a live free-generation confidence claim.
- A converted 22.37-MB F32 llama.cpp LoRA that passed real combined Mac
  acceptance on the existing 24-GB machine: both quality-passing examples
  reached Gemma and returned Routine/Priority, while Limited and Retake blocked
  the second stage.
- An earlier complete 400-image quality-only validation run through the real local upload API,
  specialist, Gemma confirmation/veto, and final policy with zero HTTP/schema
  failures. It achieved 10.75% decisive coverage and 93.02% accuracy on those
  decisions; the repeatedly viewed validation split makes this exploratory.
- A RETAKE-only factor-specific quality-attention overlay derived from the
  frozen specialist. It is explicitly labeled as technical-quality
  attention—not pathology localization—and is presentation-only: it cannot
  alter a gate.
- A visible, deterministic local decision trace showing whether the specialist
  proposed or abstained, whether Gemma confirmed or was skipped, and which
  final fail-closed policy result was released. Model prose cannot author this
  trace.
- A reproducible, veto-only DeepDRiD UWF modality experiment that caught 50/50
  validation UWF images but remains deliberately unintegrated pending a
  separately sourced multi-device check; it cannot promote a decision.

The compact quality specialist and RetinaPriority LoRA required by the demo are
versioned with checksums. The multi-gigabyte Gemma base/projector, superseded
checkpoints, evaluation outputs, and raw medical-image releases are ignored.
See [`models/README.md`](models/README.md) for the exact artifact boundary.

## Run the complete local workflow

A fresh clone includes the compact quality specialist and RetinaPriority LoRA.
Install the small web dependency set once:

```bash
cd retina-ready
python3 -m pip install -r app/requirements.txt
```

Then launch the quality specialist, validation-selected Gemma escalation LoRA,
and UI together:

```bash
./scripts/run_priority_demo.sh
```

The launcher uses the Mac-accepted exact-base Q4_0 and matching projector,
verifies the RetinaPriority LoRA checksum, then verifies the exact server alias,
absolute adapter path, and scale 1. Both services bind to loopback. The model
server's browser CORS allowlist is restricted to the local app origin.

The 14.44-GB exact-base Q4_0 and 1.19-GB projector cannot be stored in ordinary
GitHub Git objects and are not included in this repository. Before using the
full tuned route, place the hash-pinned files documented in
[`docs/LOCAL_TUNED_BUNDLE.md`](docs/LOCAL_TUNED_BUNDLE.md) at the documented
paths (or set the explicit environment-variable paths). The lightweight mode
below remains available without those base artifacts.

Open <http://127.0.0.1:8000>. The launcher keeps both services on loopback and
stops the model process it created when you press `Ctrl-C`.

The live llama.cpp adapter uses constrained free generation and is explicitly
labeled uncalibrated. It accepts only the two pinned quality-passing DeepDRiD
examples and fails closed on every identity, checksum, schema, or scope error.
The 182-image evidence described below comes from the frozen direct-logit
evaluator, not from generated-text confidence.

The earlier quality-only Gemma verifier remains reproducible through
`./scripts/run_local_demo.sh`; it is no longer the primary combined demo.

For a lightweight presentation without loading Gemma:

```bash
cd app
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## Reproduce the data and untuned fallback

```bash
./scripts/download_deepdrid.sh
python3 scripts/prepare_deepdrid.py
./ml/download_official_q4.sh
```

The last command downloads Google's frozen QAT fallback. The trained local
bundle has separate pinned provenance and checksums in
`docs/LOCAL_TUNED_BUNDLE.md`.

DeepDRiD contributes 2,000 regular fundus images:

| Split | Images | Patients | RETAKE | READY |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,200 | 300 | 624 | 576 |
| Validation | 400 | 100 | 218 | 182 |
| Test | 400 | 100 | 220 | 180 |

Integrity checks found zero missing images, duplicate image IDs, or patients
shared across splits. DeepDRiD has no official `LIMITED` label, so none was
invented for training.

## Local runtime acceptance

The final combined launcher loaded the exact-base Q4_0, matching projector, and
selected RetinaPriority LoRA from its semantic local path. Runtime checks found
one exact model alias and one exact adapter at scale 1. The four pinned cases
then produced:

| Demo case | Quality stage | Review-priority stage | Released result |
| --- | --- | --- | --- |
| Routine (`146_l2`) | READY | ROUTINE_REVIEW | Yes |
| Priority (`296_l2`) | READY | PRIORITY_REVIEW | Yes |
| Limited (`265_l2`) | LIMITED | Blocked | No |
| Retake (`431_l2`) | RETAKE | Blocked | No |

The two Gemma paths completed in about 2.3–2.7 seconds in the final acceptance;
the blocked paths did not call Gemma. This validates the local integration and
safety gates, not clinical performance. `146_l2` is a training-partition fixed
demo/calibration example and must not be described as held-out evidence.

With clean memory, the tuned exact-base Q4_0 model and GPU-offloaded projector
loaded in about 4.54 seconds on the latest retry. A four-image validation smoke
completed 4/4 requests with valid schemas, 75% accuracy, 100% RETAKE recall,
zero false READY calls, 3035.705-ms median latency, and 3281.666-ms p95 latency.
A real API request completed in about 2.2 seconds, and both the model alias and
loaded-LoRA endpoint were verified. This is an integration smoke, not a
clinical performance estimate.

The larger Q4_K_M model loaded successfully with clean memory, but its first
full-Metal vision request failed with a Metal out-of-memory error at both the
default profile and context 1024 / batch 128. A CPU-projector request completed
in 183.51 seconds, so Q4_K_M remains a slow compatibility path rather than the
live-demo configuration. Full details are in
`outputs/local-runtime-validation-20260731.json`.

The full Q4_0 hybrid run processed all 400 validation images without an HTTP or
schema failure. It returned 20 READY, 23 RETAKE, and 357 LIMITED decisions
(10.75% coverage), with 40/43 accepted decisions correct. Median API latency
was 168 ms and p95 was 5.91 seconds over all images, dominated by 356 fast
specialist abstentions. Among 44 Gemma-invoked cases, median was 5.77 seconds
and p95 was 9.26 seconds. The extra RETAKE explanation pass is included: all
23 final RETAKE decisions had a factor-specific attention map, while no READY
or LIMITED decision did. All 400 responses also carried a valid deterministic
decision trace. Every abstention stayed in the denominator; see
`outputs/hybrid-validation-exploratory.json`. Because this split was used
during development, these are exploratory reliability metrics, not a clinical
or deployment claim.

## Training reality and completed A100 run

The downloaded Q4_0 GGUF is an optimized **inference** artifact. A GGUF cannot
be trained directly with LoRA. The remote trainer loads the original Hugging
Face checkpoint, freezes the base, and trains rank-16 PEFT adapters:

```bash
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_decision_smoke.json \
  --dry-run
```

That command validates all local inputs. Gemma 4's MoE experts are raw 3-D
parameters that bitsandbytes does not convert; approximately 42.5 GiB of
experts therefore remained frozen BF16 while supported attention linears were
NF4 and 5,591,040 q/v LoRA parameters trained. This is a **partially quantized
PEFT run**, not full-model 4-bit QLoRA. Peak allocated/reserved CUDA memory was
49.49/49.63 GiB, making an 80-GB GPU the safe target.

The first full-JSON smoke learned formatting but collapsed to RETAKE. The
corrected run used exactly 64 READY and 64 RETAKE training images and one
equal-weight class token per image. In the HF PEFT runtime, its selected
step-60 adapter produced on the same 20 held-out validation images:

- 80% accuracy versus 60% for the frozen HF base and 55% for the collapsed
  adapter;
- 20/20 schema-valid JSON objects;
- 11/11 RETAKE images rejected, with zero false READY calls;
- 5/9 READY images accepted, versus 1/9 for the frozen base.

The direct decision-logit audit over all 400 validation images had ROC-AUC
0.750. A safety-first `P(READY) > 0.988` threshold was then frozen on
validation before the sealed test split was opened once. On all 400 test
images it achieved 64.25% accuracy, 61.09% balanced accuracy, 29.44% READY
recall, 92.73% RETAKE recall, a 7.27% false-READY rate, and ROC-AUC 0.756.
The test false-READY rate exceeded the validation calibration target, so these
are research-prototype results—not clinical performance claims.

The complete checksummed handoff is in `outputs/a100-handoff-20260731/`.
`docs/A100_HANDOFF.md` records the exact remote procedure and limitations.

### RetinaPriority escalation LoRA

The priority model is a separate adapter with separate manifests and labels. It
trains only on quality-passing images; ROUTINE corresponds to released DR grade
0–1 and PRIORITY to grade 2–4. Candidate selection used only the frozen 70-image
validation manifest:

| Candidate | Validation priority AUC |
| --- | ---: |
| q/v rank-16, selected step 60 | **0.92375** |
| 256-image q/v challenger | 0.89500 |
| full q/k/v/o root | 0.88000 |
| full q/k/v/o checkpoint 74 | 0.868333 |
| full q/k/v/o checkpoint 111 | 0.88000 |

Only after freezing that winner did the pipeline score the separate calibration
and evaluation partitions. On 182 evaluation images the direct logits achieved
ROC-AUC 0.956903 and balanced accuracy 0.912355 at 0.5. The calibration-frozen
selective policy releases ROUTINE only below 0.0002611903190957194 and PRIORITY
only above 0.9993736658418905; every score in between is UNCERTAIN. It accepted
65/182 evaluation images and all 65 accepted routes were correct.

The policy targets 10% patient-event risk per error type with 90% simultaneous
confidence as research evidence. The calibration cohort was too small to
certify a 5% target. None of these figures is a clinical guarantee.

Exact adapter, reports, converted LoRA, checksums, and candidate comparison are
under `outputs/a100-retinapriority-20260801/` and
`models/retinapriority-gemma4-26b/manifest.json`.

## Project map

- `app/` — local API, browser UI, model adapter, and tests.
- `data/` — dataset provenance, manifests, and local raw release.
- `demo-assets/` — attributed public sample stills and videos for an offline
  presentation.
- `ml/` — model download, server, inference, evaluation, and QLoRA utilities.
- `models/` — compact project-specific runtime artifacts plus a manifest for
  excluded multi-gigabyte base weights.
- `scripts/` — reproducible data preparation and the combined demo launcher.
- `docs/DECISIONS.md` — scope, safety, deployment, and evaluation gates.
- `docs/LOCAL_TUNED_BUNDLE.md` — tuned GGUF/LoRA provenance and local launch.
- `docs/QUALITY_SPECIALIST_MODEL_CARD.md` — specialist provenance, calibration,
  metrics, and limitations.
- `docs/EVALUATION_LEDGER.md` — split history and metric comparability.
- `docs/COMBINED_OFFLINE_EVALUATION.md` — hash-verified specialist-only offline
  quality-gate + review-priority composition over all 400 validation images.
- `docs/HACKATHON_DEMO_RUNBOOK.md` — three-minute pitch, live cases, recovery,
  and judge-safe answers.
- `docs/RECENT_RIQA_RESEARCH.md` — recent-paper synthesis and promotion rules.
- `docs/RETFOUND_GREEN_LICENSE_BLOCKER.md` — fail-closed audit explaining why
  the proposed RETFound-Green escalation challenger was not downloaded,
  trained, or integrated for this commercially funded public hackathon.

The primary success metrics are RETAKE recall, false-READY rate,
schema-valid-output rate, abstention behavior, and local median/p95 latency.
The current research-backed upgrade and its exact validation results are in
`docs/IMPROVEMENT_PLAN.md`.

Project code is licensed under the MIT License. Dataset samples, video clips,
pretrained weights, and Gemma-derived artifacts retain their upstream terms;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Historical frozen-QAT baseline

A deterministic, class-balanced 12-image validation smoke test produced:

- 12/12 schema-valid responses.
- 6/6 RETAKE images rejected; zero false-READY predictions.
- 1/6 READY images accepted; the other five were conservatively rejected.
- 7/12 overall accuracy, 3.98-second median latency, and 6.33-second p95.

This is a pipeline smoke test, not a publishable performance estimate. It
shows the untuned fallback's measured failure mode clearly: safe but
over-rejecting.
The report is saved locally at `outputs/baseline-val-12.json`; a full
patient-held-out evaluation is required before making performance claims.

```bash
python3 ml/evaluate_local.py \
  --limit 12 --sampling stratified --seed 42 \
  --summary-only --output outputs/baseline-val-12.json
```
