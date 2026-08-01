# RetinaReady + RetinaPriority hackathon demo runbook

## One-sentence product

RetinaReady checks whether a fundus image is usable while the patient is still
present; only usable images reach RetinaPriority, a separate local Gemma 4 LoRA
that suggests routine or priority clinician review without making a diagnosis.

## The three-minute demo

### 0:00–0:25 — The wasted visit

> Retinal screening has two different failure points. First, a technically bad
> image may reach a grader after the patient has left. Second, a usable image
> that deserves faster attention may sit in the normal queue. We close both
> loops locally: capture quality first, review priority second.

Point to the **Local only** indicator. The browser, 33 MB quality specialist,
and one Gemma server communicate over numeric loopback addresses. Images are
processed in memory and are not intentionally uploaded or persisted.

### 0:25–0:50 — Guided acquisition replay

Press **Start replay**. Say:

> This is a clearly labelled retrospective DeepDRiD replay—not a live patient
> camera. Lightweight local frame telemetry follows field position, focus,
> illumination, and stability at six checks per second. Once five frames are
> consistently usable, it autocaptures once and sends the untouched still into
> the real quality-first workflow. Gemma does not run in the preview loop.

Point to the single changing instruction and autocapture-lock meter. The
replay should move through centering, stability, focus or illumination, then
freeze the original `296_l2` frame. On the final launcher, that exact frame
passes RetinaReady and reaches the real local Gemma priority LoRA.

Do not call the replay live patient acquisition, device validation, learned
movement guidance, or pathology detection.

Optional judge follow-up: press **Open camera recording** and select
`data/external/fundus-video/ultra-wide-field-video-fundus-photography.mp4`.
Say:

> This is a real, openly licensed moving color retinal recording from a
> purpose-built clinical camera. We decode it locally and run the same
> six-hertz technical telemetry. After five stable frames, we freeze one JPEG
> locally and run it through quality first, then review priority only if it is
> usable. The raw video never enters either model. This candidate route is an
> explicitly enabled experimental OOD demo, not device-video validation.

For a visible fail-closed example, load
`data/external/fundus-video/kestrel-3100m-self-service.mp4`. It shows a real
tabletop Kestrel workflow but does not expose a raw color retinal feed, so the
preview should remain at **No color fundus field detected** and finish with no
submitted still. Do not use either recording to claim device validation or
model accuracy. The retrospective replay remains the dependable main-stage
path because only its untouched pinned still is inside the verified fixed-hash
model contract; the extracted video still is intentionally labelled experimental.

### 0:50–1:35 — Fixed dataset cases

Choose **Combined** and use only the four labeled buttons.

1. **Routine:** the quality gate returns `READY`; the Gemma LoRA returns
   `ROUTINE_REVIEW`. Say: “Usable image, normal human-review queue.” This
   `146_l2` training-partition case is a fixed calibration/demo example, not
   held-out evidence.
2. **Priority:** the same quality gate returns `READY`, but RetinaPriority
   returns `PRIORITY_REVIEW`. Say: “Usable image, faster clinician review—not a
   diagnosis.” This is the fixed `296_l2` validation example.
3. **Limited:** the quality specialist abstains, so the 26B call is blocked.
   Say: “Uncertainty is a product action, not a crash.”
4. **Retake:** the quality gate blocks prioritization and gives immediate
   recapture guidance. Call the optional overlay **technical-quality
   attention**, never pathology localization.

The visible trace should show `Quality gate → Review priority → Safety policy`.
For Routine and Priority, the first two stages complete and the review route is
released. For Limited and Retake, the second stage is visibly blocked and no
route is released.

### 1:35–2:10 — Why the two-model harness wins

> We did not force one model to solve two different problems. A compact retinal
> specialist handles low-level acquisition quality quickly. Only exact READY
> images reach a separately trained Gemma 4 escalation LoRA. That saves compute,
> prevents unusable images from receiving confident queue labels, and lets us
> evaluate each safety boundary independently.

Application code owns the gate, normalized wording, and fail-closed behavior.
The live Gemma output is constrained to one of two exact JSON contracts. Any
wrong image, checksum, alias, adapter path, scale, schema, or server state
becomes `UNCERTAIN`.

### 2:10–2:45 — Evidence, stated precisely

> We selected the winning LoRA only on a frozen 70-image validation set. On a
> separate patient-disjoint evaluation of 182 technically usable images, its
> direct decision logits achieved ROC-AUC 0.956903 and balanced accuracy
> 0.912355 at the ordinary 0.5 threshold. We then froze a conservative
> three-way policy on a separate calibration partition. It confidently routed
> 65 of the 182 evaluation images—36 Priority and 29 Routine—and all 65 routes
> were correct; the other 117 were explicitly Uncertain.

Immediately qualify it:

> Those are offline direct-logit research results, not clinical validation.
> The live llama.cpp demo uses constrained free generation, so it is labeled
> uncalibrated and does not claim those thresholds as live confidence.

The calibration set supported a research target of 10% patient-event risk per
error type at 90% simultaneous confidence. It was too small to certify 5%; do
not round this into a clinical guarantee.

### 2:45–3:00 — Close

> The differentiator is the handoff: fix unusable images now, prioritize usable
> images locally, and make uncertainty visible instead of manufacturing a
> diagnosis. Patient data never needs to touch a cloud API.

## Start and verify before presenting

Run:

```bash
cd "/Applications/Personal App/discord-mutual-friends-and-servers-main/retina-ready"
./scripts/run_priority_demo.sh
```

Open <http://127.0.0.1:8000>. Click Routine, Priority, Limited, and Retake once
before judges arrive, then run **Start replay** through autocapture once. The
launcher verifies the LoRA checksum and then verifies the exact server alias,
absolute adapter path, and scale 1.

Health check:

```bash
curl -fsS http://127.0.0.1:8000/api/health | jq .
```

Expected top-level fields include `status: ready`, `mode: specialist-local`,
`model_verified: true`, `specialist_verified: true`, `privacy: local-only`, and
`network_required: false`. The nested `escalation` object must report:

- `status: ready`
- `profile: gemma-lora-free-generation-uncalibrated-experimental`
- `model_verified: true`
- `lora_verified: true`
- `adapter_hash_verified: true`
- `release_enabled: true`
- `input_scope: fixed-deepdrid-quality-pass-demo-samples`
- `clinical_use: false`

Do not present if those nested identity checks are false.

## Emergency fallback: historical compact-priority path

If the 26B server cannot load, use the real two-specialist fallback rather than
deterministic presentation output. This preserves the product-shell demo, but
it is not the final selected Gemma architecture or its evidence path:

```bash
cd "/Applications/Personal App/discord-mutual-friends-and-servers-main/retina-ready/app"
RETINA_ANALYZER=specialist \
RETINA_SPECIALIST_DEVICE=cpu \
RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1 \
RETINA_ESCALATION_ENGINE=specialist \
RETINA_ESCALATION_DEVICE=cpu \
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Say explicitly that this fallback uses two compact specialists, not Gemma. It
still demonstrates the quality-first harness and remains fully local.

## Failure recovery

- If the model is slow, show Limited or Retake first; both short-circuit before
  Gemma.
- If a Gemma request or identity check fails, point to the `UNCERTAIN` result as
  the intended fail-closed action, then use the compact fallback.
- If another service occupies port 8082, stop it. The launcher refuses to reuse
  a healthy server with the wrong alias or adapter.
- If browser animation is interrupted, stop the replay and use the Priority
  sample directly. Both paths submit the same untouched pinned still.
- Do not switch to arbitrary images on stage. The live research path is
  intentionally pinned to the four DeepDRiD examples, and only the two
  quality-passing examples may reach Gemma.

## Judge questions

**Is Priority a diagnosis?**

No. It is a queue suggestion under the dataset's declared referable-DR
threshold. A clinician still reviews every image. The UI never names a disease,
declares an eye healthy, recommends treatment, or delays review.

**Why not use Gemma for image quality too?**

The compact specialist is faster and stronger at acquisition-specific signals.
The architecture reserves Gemma for the semantically richer review-routing
stage and avoids spending 26B inference on images that should be retaken.

**Why is 65/65 not a claim of 100% accuracy?**

Because the frozen policy abstained on 117/182 evaluation images. The correct
statement is 35.7% coverage with 65/65 accepted routes correct on this held-out
research partition. Broader clinical validation is still required.

**Why do the offline metrics not appear as live confidence?**

The measured evaluator compares the two class logits directly in the Hugging
Face PEFT runtime. llama.cpp serves constrained free generation from the
converted LoRA. They demonstrate the same trained task but are different
inference contracts; we refuse to pretend generated labels are calibrated
probabilities.

**Why on-device instead of a cloud API?**

It returns capture guidance before the patient leaves, works during poor
connectivity, avoids per-image cloud latency and cost, and keeps identifiable
retinal pixels off a third-party API. Local execution reduces exposure; it does
not by itself establish HIPAA compliance.

**Why did the smaller training run beat the larger ones?**

Validation decided, not parameter count. The q/v rank-16 step-60 adapter scored
0.92375 AUC. The 256-image q/v challenger scored 0.895; the full q/k/v/o root
and checkpoints 74/111 scored 0.880, 0.868333, and 0.880. Calibration and
evaluation were not consulted until after that validation-only selection was
frozen.

## Evidence files

- `models/retinapriority-gemma4-26b/manifest.json` — selected artifact, hashes,
  candidate comparison, evaluation, and policy summary.
- `outputs/a100-retinapriority-20260801/selected-smoke-adapter/` — exact PEFT
  adapter and validation/calibration/evaluation reports.
- `outputs/a100-retinapriority-20260801/integrity/with-challenger-v1.comparison.json`
  — remote/local artifact comparison.
- `docs/GEMMA_ESCALATION_QUALITY_PASS.md` — dataset and training contract.
- `docs/GEMMA_ESCALATION_POSTTRAIN.md` — immutable selection and evaluation
  automation.
- `docs/COMBINED_WORKFLOW.md` — typed quality-first product contract.
- `docs/EVALUATION_LEDGER.md` — split history and claim boundaries.
