# RetinaReady local demo

A dependency-light FastAPI shell and responsive single-page interface for the
RetinaReady + RetinaPriority demo. The accepted demo path runs the compact,
exact-hash quality specialist first; only an exact `READY` result reaches the
validation-selected q/v rank-16 step-60 Gemma escalation LoRA. All browser
assets are local.

The interface exposes three product modes: **Quality only**, **Escalation
only**, and **Combined**. Quality only and Escalation only remain useful test
surfaces; the final demo uses Combined. The existing `POST /api/analyze`
quality contract is unchanged. `POST /api/workflow` adds fail-closed
orchestration via the `X-Product-Mode` header. In Combined mode, only an exact
quality `READY` result can reach review prioritization. See
`docs/COMBINED_WORKFLOW.md` for the typed decision and adapter contract.

An earlier compact review-priority specialist remains available only as a
historical low-memory presentation fallback. It is not the selected final
RetinaPriority path; its separate promotion manifest and unchanged
`runtime_integration_authorized=false` and `diagnostic_use_authorized=false`
source flags keep that boundary explicit.

## Guided acquisition replay

The capture card now includes **Start replay**, a labelled prototype of
continuous color-fundus acquisition guidance. It renders the fixed DeepDRiD
Priority sample as a moving preview and analyzes reduced-resolution canvas
frames locally at about 6 Hz. A dependency-free controller measures technical
field centering, edge-based focus, illumination, and frame-to-frame stability,
then displays one smoothed instruction at a time.

After five consecutive passing ticks, the controller freezes the best-known
state and sends the original, byte-for-byte DeepDRiD file through the existing
`POST /api/workflow` combined path. The transformed preview frames never enter
the quality specialist or Gemma. This preserves both exact-hash allowlists and
keeps Gemma out of the frame loop.

The interface identifies this as a simulated retrospective replay and a local
technical prototype—not a live patient camera, learned frame-level clinical
model, or validated device controller. See
`docs/VIDEO_INFERENCE_ROADMAP.md` for the implemented boundary and the path to
real device-video validation.

The adjacent **Open camera recording** control is a separate experimental
preview. It accepts MP4, WEBM, or MOV input up to 128 MB, renders the decoded
frames into the same local telemetry canvas, and applies a conservative
reference-relative color-field presence guard. A recording with no visible
color retinal field cannot accumulate the five passing frames required for a
candidate. Duplicate or stalled media timestamps are not counted as stable
frames.

This real-recording path deliberately stops at “candidate frame.” Video bytes
and canvas frames never become the still-image `state.file`, never call
`POST /api/workflow`, and never enter either trained model. An operator must
export a separate still from the fundus camera before the existing
quality-first harness can run. The app includes two CC BY 4.0 QA inputs under
`data/external/fundus-video/`: a genuine moving PedCam color-fundus video for a
positive loader test and a Kestrel 3100m tabletop-workflow recording for the
fail-closed no-field test. See that folder's `ATTRIBUTION.md` for provenance
and hashes.

## Run the final combined demo

From the repository root, launch the quality specialist, pinned Gemma runtime,
selected escalation LoRA, and app together:

```bash
./scripts/run_priority_demo.sh
```

Open <http://127.0.0.1:8000>. The interface exposes four sample buttons:
**Routine**, **Priority**, **Limited**, and **Retake**. Routine is `146_l2` from
the training-split calibration slice and is a calibration/demo example, not a
held-out result. Priority is the fixed `296_l2` validation example and retains
the backwards-compatible `READY` sample API key. Limited and Retake stop after
the quality stage; only Routine and Priority may reach Gemma.

For app-shell development without starting the final Gemma path:

```bash
cd retina-ready/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Deterministic mode keeps the controls clearly labelled as presentation inputs
rather than model evidence.

## Run the compact quality specialist (no Gemma server)

The 33 MB frozen quality bundle can run by itself on the 24 GB Mac. It does not
load the 26B model, start a model server, require a network connection, or use
`GEMMA_API_URL`:

```bash
cd retina-ready/app
export RETINA_ANALYZER=specialist
export RETINA_SPECIALIST_DEVICE=cpu
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

This profile requires the existing local `torch`, `torchvision`, `numpy`, and
Pillow runtime in addition to `app/requirements.txt`. At startup and before
each decision, it checks independent code-pinned SHA-256 values for the bundle
manifest, DenseNet backbone, and both frozen heads. The colocated manifest is
not its own trust root.

The quality specialist has no independently validated modality/OOD detector,
so this safe live profile intentionally accepts only the four exact DeepDRiD
images behind the ROUTINE, READY, LIMITED, and RETAKE sample API keys. The UI
labels READY as **Priority** while preserving that existing key. The ROUTINE
image is the training-split calibration/demo example described above, not a
held-out example. Any other bytes, including a decodable non-fundus image or a
different dataset image, return `LIMITED` as outside the demo scope before
inference. Missing dependencies, missing or changed files, malformed output,
decode errors, and inference errors also fail closed to `LIMITED`.
`/api/health` reports `mode: specialist-local`,
`profile: quality-specialist`, `model_verified: true`,
`specialist_verified: true`, `privacy: local-only`, and
`network_required: false` only when that exact bundle is available; its
`input_scope` is `fixed-deepdrid-demo-samples`.

## Historical compact-priority fallback

The earlier two-compact-specialist presentation path can still be reproduced
without Gemma by adding the separate nonclinical review-priority opt-in before
starting the app:

```bash
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_DEVICE=cpu
```

Then choose **Combined** and use the fixed dataset buttons. This fallback still
enforces the quality-first gate, but it is not the final selected Gemma
architecture or evidence path.

## Final Gemma RetinaPriority path

The raw app default retains `RETINA_ESCALATION_ENGINE=specialist` for backward
compatibility, but the accepted launcher explicitly selects `gemma`. It keeps
`RETINA_ANALYZER=specialist` for the quality gate and starts one llama.cpp
server with the exact base, projector, and selected escalation LoRA. The server
is bound to loopback, uses a distinct exact alias, and receives the LoRA by
absolute path.

The accepted bundle is already wired into a checksum- and identity-verifying
launcher. From the repository root, prefer:

```bash
./scripts/run_priority_demo.sh
```

The manual commands below document the same contract for debugging.

```bash
llama-server \
  --host 127.0.0.1 --port 8082 \
  --model /absolute/path/to/gemma-4-26b-q4_0.gguf \
  --mmproj /absolute/path/to/gemma-4-mmproj-bf16.gguf \
  --lora /absolute/path/to/retinapriority-lora-f32.gguf \
  --alias retinapriority-gemma4-26b
```

Start the app separately:

```bash
cd retina-ready/app
export RETINA_ANALYZER=specialist
export RETINA_SPECIALIST_DEVICE=cpu
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_ENGINE=gemma
export RETINA_ESCALATION_GEMMA_API_URL=http://127.0.0.1:8082
export RETINA_ESCALATION_GEMMA_MODEL_ID=retinapriority-gemma4-26b
export RETINA_ESCALATION_GEMMA_LORA_PATH=/absolute/path/to/retinapriority-lora-f32.gguf
export RETINA_ESCALATION_GEMMA_LORA_SHA256=<exact-lowercase-sha256>
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Before every inference, the adapter re-hashes the LoRA, requires `/health` to
be ready, requires the exact alias from `/v1/models`, and requires exactly the
configured absolute LoRA path at scale 1 from `/lora-adapters`. It accepts only
the exact hashes of two quality-passing DeepDRiD demo images: the
training-split `146_l2` ROUTINE calibration/demo example and the existing
READY-keyed `296_l2` PRIORITY validation example. The routine example is not
held-out evidence. Its output is decoding-constrained by a strict JSON schema
and must match the training target; only `ROUTINE` and
`PRIORITY` map to the two review queues. Every other input, identity, schema,
hash, or loopback-server outcome returns `UNCERTAIN` with no queue release.

This is the validation-selected q/v rank-16 step-60 adapter. In the separate
182-image direct-logit evaluation it achieved ROC-AUC 0.956903 and balanced
accuracy 0.912355 at threshold 0.5; the calibration-frozen selective policy
accepted 65/182 images, and all 65 accepted routes were correct. Those are
offline Hugging Face decision-logit results.

The live llama.cpp path is **uncalibrated experimental free generation** and
is pinned to the two quality-passing demo inputs described above. Its generated
labels do not inherit the direct-logit thresholds, coverage, or confidence
claims. Runtime confidence stays null, and the path is not clinically
validated.

## Historical: earlier quality-confirmation Gemma profile

The following profile used Gemma to confirm or veto quality decisions. It is
retained for reproduction and is not the final RetinaPriority escalation demo.

Point the existing analyzer boundary at a local llama.cpp-style,
OpenAI-compatible multimodal server:

```bash
export RETINA_ANALYZER=local
export GEMMA_API_URL=http://127.0.0.1:8081
export MODEL_ID=retinaready-gemma4-26b
export RETINA_HYBRID=1
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_DEVICE=cpu
uvicorn main:app --host 127.0.0.1 --port 8000
```

`GEMMA_API_URL` also accepts a URL ending in `/v1` or
`/v1/chat/completions`. In `auto` mode (the default), setting
`GEMMA_API_URL` selects the local model; otherwise the deterministic demo
engine is used. Non-loopback model URLs are rejected unless
`RETINA_ALLOW_REMOTE_MODEL=1` is explicitly set.

The escalation opt-in is independent from Gemma and requires no network. The
adapter reads
`models/retinaready-escalation-demo/promotion-manifest.json`, re-verifies its
bound artifact, report, and backbone before every release attempt, and maps
the specialist's internal `ROUTINE` / `PRIORITY` states to
`ROUTINE_REVIEW` / `PRIORITY_REVIEW`. Equality or a score between the strict
thresholds returns `UNCERTAIN`. Do not set the opt-in outside the local
hackathon research demonstration.

The browser sends raw image bytes, `Content-Type`, and `X-Filename`; no
multipart parser or cloud upload is needed. All analyzer profiles use the same
`POST /api/analyze` response shape.

The hybrid adds the 33 MB multi-task quality specialist before Gemma. A
patient-level selective policy owns the READY/RETAKE/LIMITED gate; Gemma can
confirm or veto a decision but cannot promote LIMITED. A policy-level LIMITED
case skips the 26B call entirely. A Gemma timeout, model disagreement, invalid
image, or malformed response fails closed to LIMITED rather than returning a
500 or a browser-invented live result. The combined launcher uses a 30-second
server timeout and the browser aborts after 35 seconds.

The browser response keeps the same status field. The deterministic demo can
return:

- `READY`
- `LIMITED`
- `RETAKE`
- `UNSUPPORTED`

The local model contract is stricter: `READY`, `LIMITED`, or `RETAKE`.
Unsupported modalities and uncertain assessments normalize to `LIMITED`;
invalid model output also fails safely to an uncertain `LIMITED` result.
Confidence is either `null` or 0–1. Factor scores are 0–100 and are omitted
when unavailable or when the result is `LIMITED`.

`analyzer.py` converts that contract into the existing browser `issues`,
`instruction`, `scores`, and runtime `meta` fields. Keep the result restricted
to technical image quality—never diagnosis or treatment guidance.

## Historical quality-model runtime acceptance

The earlier accepted 24-GB Mac quality profile used the exact-base Q4_0 GGUF,
matching BF16 projector, and F32 RetinaReady quality LoRA. Local llama.cpp build 10210 loaded that
profile in about 4.54 seconds on the latest clean retry. A four-image validation
smoke completed 4/4 requests with valid schemas, 75% accuracy, 100% RETAKE
recall, zero false READY calls, 3035.705-ms median latency, and 3281.666-ms p95
latency. A separate real API request completed in about 2.2 seconds. The
`retinaready-gemma4-26b` API alias and loaded-LoRA endpoint were verified.

Those four images establish integration readiness, not clinical performance.
The higher-bit Q4_K_M alternate is not supported for full-Metal vision on this
24-GB Mac: it loaded under two clean-memory profiles, but both first vision
requests failed with a Metal out-of-memory error. Its CPU-projector path took
183.51 seconds and is retained only for compatibility testing.

Uploads are limited to 16 MB and to JPEG, PNG, or WEBP, processed in memory,
and sent only to the loopback model URL unless remote access is explicitly
enabled. Unsupported or uncertain images and malformed model output fail
closed to `LIMITED`. The
standalone deterministic mode is clearly labelled and must not be presented as
model analysis of an uploaded image.

The specialist sigmoid is an uncalibrated decision score, so both specialist
and hybrid modes leave
`confidence` null. See `docs/QUALITY_SPECIALIST_MODEL_CARD.md` for the exact
patient splits, thresholds, hashes, assumptions, and exploratory metrics. The
current patient-level bounds are explicitly post-hoc because the DeepDRiD
patients influenced earlier experiments; they are not a deployment guarantee.

The three-case hybrid acceptance completed READY in 3.15 seconds, LIMITED in
0.21 seconds, and RETAKE in 3.49 seconds. A subsequent run through the real API
covered all 400 validation images with zero HTTP/schema failures, 10.75%
decisive coverage and 93.02% accepted accuracy. Overall median/p95 latency was
168 ms/5.91 s, dominated by 356 specialist short-circuits; the 44 Gemma-invoked
cases measured 5.77 s/9.26 s. The refreshed run included the explanation pass:
all 23 final RETAKE decisions had a quality-attention PNG and no READY or
LIMITED decision did. All 400 responses had a schema-valid decision trace. The
reports are
`outputs/hybrid-runtime-acceptance-20260731.json` and
`outputs/hybrid-validation-exploratory.json`. The validation split was already
viewed during development, so this is exploratory reliability evidence rather
than a clinical performance estimate.

RETAKE responses may also include a local factor-specific quality-attention PNG
for the specialist's weakest acquisition factor. READY and LIMITED skip this
pass so a visualization never implies a defect or over-explains uncertainty.
The browser labels it
“Model quality attention — not pathology localization.” It runs only after the
unchanged decision pass, is omitted if generation fails, and cannot promote or
otherwise alter a READY/LIMITED/RETAKE result. The visualization has not been
validated as a lesion map or clinical explanation.

## Verify

```bash
python3 -m compileall -q .
node --check static/app.js
python3 -m unittest discover -s tests -v
```
