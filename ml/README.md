# RetinaReady Gemma 4 model path

The specialist-only offline quality-gate + review-priority evaluator is documented in
[`docs/COMBINED_OFFLINE_EVALUATION.md`](../docs/COMBINED_OFFLINE_EVALUATION.md).
It reuses hash-verified 400-image score reports, runs no new inference, retains
every blocked/uncertain image in the denominator, and writes
`outputs/combined-offline-evaluation/report.json`.

## Decision

Gemma 4 **26B A4B Q4 inference is feasible but tight** on this 24-GB M5 Pro.
The primary tuned bundle uses an exact-base Q4_0 model, matching BF16
projector, and separate F32 LoRA; the same exact base is also retained as a
higher-bit Q4_K_M alternate. Gemma 4 **26B training is not feasible on
this Mac**. Run the Transformers/TRL PEFT path on a Linux NVIDIA host with an
A100/H100 80 GB.
The measured peak for the supplied dtype-preserving one-step probe was about
49.4 GiB, so a nominal 48-GB GPU is below the demonstrated requirement and
leaves no safe runtime headroom.

The exact official model is `Gemma 4 26B A4B`, a mixture-of-experts model
with 26B total parameters and about 4B activated per token. All experts
still have to be held in memory.

| Path | Official static-weight estimate | This machine |
| --- | ---: | --- |
| 26B BF16 inference | 57.7 GB | No |
| Tuned 26B Q4_0 inference | 14.44 GB base, before projector/KV/runtime | Primary 24-GB path |
| Tuned 26B Q4_K_M inference | 16.80 GB base, before projector/KV/runtime | CPU-projector alternate |
| Untuned QAT 26B Q4_0 inference | 14.44 GB base, before projector/KV/runtime | Yes; fallback |
| 26B QLoRA training | Far above inference footprint | No; CUDA cloud GPU |
| E2B Q4_0 inference | 2.9 GB | Safe fallback |

The default tuned inference set totals 15.66 GB: a 14.44-GB Q4_0 base,
1.19-GB image projector, and 22.4-MB F32 LoRA. On a unified-memory Mac, the
OS, display, Metal runtime, image embeddings, and KV cache share the remaining
memory. The launcher therefore pins context to 2048, batch/micro-batch to
512, and parallelism to one.

## 1. Tuned bundle and official fallback

The tuned exact-base bundle is recorded in `docs/LOCAL_TUNED_BUNDLE.md` and is
selected automatically when complete. The following download is only for the
separate **untuned QAT fallback**.

The Google repository is public and Apache-2.0 licensed; it did not
require authentication when checked on 2026-07-30. `HF_TOKEN` is only
needed if Hugging Face rate-limits the request.

```bash
cd retina-ready
./ml/download_official_q4.sh
```

The script pins revision `d1c082be9cf3c8a514acf63b8761f4b41935842e`
and downloads only:

- `gemma-4-26B_q4_0-it.gguf`
- `gemma-4-26B-it-mmproj.gguf`

Do not download the 51.6-GB BF16 checkpoint onto this Mac. Also do not
quantize it locally: Google already provides the official
quantization-aware-trained Q4_0 GGUF. It remains separate from the tuned
standard-base bundle and is never used with the RetinaReady LoRA.

## 2. Run fully local inference

Install current llama.cpp and launch the bounded-memory profile:

```bash
brew install llama.cpp
cd retina-ready
./ml/serve_local.sh
```

The launcher disables the reasoning channel for this bounded structured-output
task and waits for `/health` before reporting readiness. In another terminal:

```bash
cd retina-ready
python3 ml/infer_local.py path/to/fundus.jpg
```

The request uses a base64 data URL and `127.0.0.1`; the image never needs
to leave the laptop. The strict response contract is:

```json
{
  "decision": "READY | LIMITED | RETAKE",
  "confidence": null,
  "issues": ["artifact | blur | field_cutoff | unsupported_modality | uncertain"],
  "scores": {
    "artifact": 0,
    "clarity": 0,
    "field_definition": 0
  },
  "retake_instruction": null,
  "disclaimer": "Technical image-quality assessment only; not a diagnosis."
}
```

Every score uses 0-100 with 100 meaning best technical quality.
`infer_local.py` validates and normalizes the object. Invalid model
output fails closed as `LIMITED`.

If the 26B model causes memory pressure, reduce GPU layers or move only the
vision projector to CPU:

```bash
RETINA_READY_GPU_LAYERS=40 ./ml/serve_local.sh
RETINA_READY_MMPROJ_OFFLOAD=off ./ml/serve_local.sh
```

The CPU-projector path is slow but completed a real request. The safer event
fallback remains the official smaller E2B/E4B QAT model if the 26B path cannot
meet the target device's latency budget.

## 3. Validate the DeepDRiD training input locally

The manifests are patient-disjoint. The specialist does not use the test set;
the test was opened once only for the earlier frozen Gemma smoke evaluation:

```bash
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json \
  --dry-run
```

Expected inputs:

- `data/manifests/train.csv`: 1,200 images / 300 patients
- `data/manifests/val.csv`: 400 images / 100 patients
- `data/manifests/test.csv`: historical one-time Gemma evaluation only

DeepDRiD supplies an official binary overall-quality label. It does not
supply a `LIMITED` class or official factor cutoffs. The trainer keeps
the binary READY/RETAKE target, preserves factor ordering as 0-100
scores, and documents its issue tags as a UI heuristic:

- artifact raw score >= 4 -> `artifact`
- clarity raw score <= 6 -> `blur`
- field-definition raw score <= 6 -> `field_cutoff`

Do not present those thresholds as clinical ground truth.

The separate review-priority task must operate only after this quality gate.
Its deterministic quality-pass manifests, lineage hashes, counts, and new full
Gemma config are documented in
[`docs/GEMMA_ESCALATION_QUALITY_PASS.md`](../docs/GEMMA_ESCALATION_QUALITY_PASS.md).

## 4. Prepare 26B QLoRA on a cloud GPU

Use Linux plus an NVIDIA BF16 GPU. The practical requirement is an A100/H100
80 GB: the measured 49.4-GiB peak already exceeds a nominal 48-GB card before
safe runtime headroom. An L4's 24 GB is suitable for the
official small-model tutorial but is too risky for this 26B multimodal
job.

Before running any trainer on a new host, copy the project and extracted
dataset to the GPU host and run the read-only preflight:

```bash
cd retina-ready
python3 ml/preflight_a100.py \
  --config ml/configs/gemma4_26b_smoke.json \
  --json-output ml/runs/a100-preflight.json
```

It checks Linux/Python/disk (150 GiB free by default), CUDA GPU name and VRAM, BF16 support, dependency
imports and versions, all train/validation/test manifests and images, patient
leakage, and pinned Hugging Face revision/config access. Hugging Face access is
limited to metadata and small JSON config files; it never downloads model weights.
The current repositories allow anonymous config access, so a token is optional;
the check also validates a cached or `HF_TOKEN` credential when one is present.
Exit `0` means ready and exit `1` means at least one blocker remains. To build
a fresh Linux virtual environment and then run that same preflight:

```bash
./ml/bootstrap_a100.sh
```

Neither command imports or launches `train_qlora.py`.

The canonical future training sources are pinned:

- `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`

The following commands are intentionally not run by any setup script:

```bash
cd retina-ready
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r ml/requirements-train.txt

# First after approval: one image and one optimizer step to measure peak VRAM.
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_vram_probe.json

# Historical first smoke: bounded 128-image / 60-step end-to-end proof.
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json

# Corrected classifier smoke: balanced rows and one equal-weight decision token.
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_decision_smoke.json

# Continue only if generated-output metrics beat the frozen baseline.
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_full.json
```

The one-step probe and both 26B configs enforce the 48 GiB minimum; an A100
40 GB cannot bypass that guard through configuration. Every future run writes
an atomic `run_provenance.json` containing revisions, effective configuration,
manifest/selection hashes, package/GPU versions, metrics, failure state, and
peak CUDA memory. Resume is always explicit:

```bash
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json \
  --resume-from-checkpoint ml/runs/gemma4-26b-retina-smoke/checkpoint-40
```

A fresh run refuses to reuse a non-empty output directory, and a resume
checkpoint must be directly inside the configured output directory. This
protects earlier adapters and provenance from accidental overwrite.

The training model is the pinned BF16 instruction checkpoint
`google/gemma-4-26B-A4B-it`. Bitsandbytes quantizes supported 2-D linear
attention weights to NF4, but Gemma 4's MoE experts are raw 3-D parameters and
are not converted by bitsandbytes. The trainer therefore freezes the base
without PEFT's generic FP32 upcast: the approximately 42.5-GiB expert block
stays frozen in BF16 while rank-16 LoRA adapters train on language q/v
projections. This is a reproducible **partially quantized PEFT path**, not a
claim that every 26B parameter is 4-bit. The processor is explicitly
`google/gemma-4-E2B-it`, which supplies the shared Gemma 4 chat template. The
trainer verifies that its vocabulary and multimodal token IDs match the model.

The collator verifies that the rendered generation prompt is an exact token
prefix, then masks system, user, image, padding, and assistant-header tokens.
Two explicit loss scopes are available. `assistant` supervises the full JSON;
`decision_token` keeps that JSON as causal context but supervises only the
first distinguishing READY/RETAKE value token. With stratified sampling this
gives each image and class equal loss weight, preventing longer RETAKE JSON
targets from dominating the small smoke run.

The adapter deliberately does not train or duplicate `lm_head` and
`embed_tokens`: the pinned model and processor vocabularies and special-token
IDs already match, no tokens are added, and saving those tied parameters would
add hundreds of millions of trainable values unrelated to the quality
boundary. This path is intentionally different from Google's local QAT GGUF:

- The existing QAT Q4_0 model remains the optimized untuned fallback.
- Bitsandbytes NF4 + PEFT was the trainable CUDA path.
- The adapter targets the standard BF16 checkpoint and must never be applied
  to Google's separately trained QAT checkpoint.
- The tuned local runtime uses fresh Q4_0 and Q4_K_M conversions of that exact
  standard base, its matching BF16 projector, and the LoRA as a separate F32
  GGUF passed to `llama-server --lora`.

The separate adapter is deliberate. A numerical merge audit found that adding
these small LoRA deltas directly to BF16 q/v weights rounded roughly 44-56% of
individual delta entries to zero and materially changed last-token logits;
quantizing that merged result would lose still more of the learned signal.
Keeping the 22.4-MB adapter in F32 lets llama.cpp apply it after dequantizing
the exact matching Q4 base. This is still fully offline local inference.

When the tuned projector, LoRA, and either exact-base quantization are present
in `models/retinaready-gemma4-26b-tuned/`, `ml/serve_local.sh` selects them
automatically and prefers Q4_0. If that directory is absent or incomplete, it
safely falls back to the official untuned QAT bundle. See
`docs/LOCAL_TUNED_BUNDLE.md` for the pinned revisions, checksums, conversion
details, and explicit overrides.

## 5. Smaller fallback

The exact same trainer supports:

```bash
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_e2b_fallback.json
```

Google's official vision QLoRA guide explicitly supports E2B and calls
for a BF16-capable NVIDIA GPU with more than 16 GB. This is the model to
tune in a future iteration if the 26B bundle cannot meet a target device's
memory or latency budget.

## Compact multi-task quality specialist

The live hybrid uses a frozen ImageNet DenseNet-121 backbone plus a five-member
MLP ensemble trained on DeepDRiD's overall quality and three acquisition
factors. Reproduce the frozen protocol with:

```bash
python3 ml/train_quality_specialist.py \
  --device cpu \
  --auxiliary-weight 1.0 \
  --output-dir outputs/quality-specialist-rigorous-factors
```

The trainer creates patient-disjoint fit, tuning, and calibration partitions.
It selects stopping epochs on tuning patients, refits on fit+tuning, freezes the
model, and only then computes patient-level READY and RETAKE thresholds. Delta
is 0.025 per gate; on a genuinely fresh exchangeable calibration cohort, that
would give at least 95% simultaneous confidence by a union bound. The current
DeepDRiD threshold set is historically reused, so its bounds are explicitly
nominal/post-hoc rather than a deployment guarantee. The raw sigmoid remains
an uncalibrated decision score. The trainer refuses `test.csv`.

See `docs/QUALITY_SPECIALIST_MODEL_CARD.md` for exact hashes, split counts,
thresholds, exploratory metrics, and limitations.

Run the frozen, no-tuning MSHF device-shift evaluator with:

```bash
python3 ml/evaluate_external_mshf.py
```

It verifies the 1.1 GB archive checksum, reconstructs majority-vote labels from
the three released annotators, evaluates only the authors' 260-image test
directory, and reports results by source and camera class. This MSHF test has
now been opened; future changes must not be selected against it.

With the complete local hybrid already running, exercise the actual upload API
over every validation image with:

```bash
python3 ml/evaluate_hybrid_runtime.py \
  --output outputs/hybrid-validation-exploratory.json
```

The evaluator refuses the official test manifest, verifies the local-only
hybrid/model/LoRA/specialist health identity, and retains every LIMITED,
timeout, non-200 response, and malformed response in the denominator. The
recorded 400-image run had zero HTTP/schema failures, 10.75% coverage, 93.02%
accepted accuracy, 168-ms median API latency, and 5.91-s p95 latency. The
all-image latency is dominated by 356 specialist short-circuits; the 44
Gemma-invoked cases had 5.77-s median and 9.26-s p95 latency. The evaluator
also verifies that all 23 final RETAKE decisions—and no READY or LIMITED
decisions—contain the optional local quality-attention map, and that all 400
responses carry a consistent application-authored decision trace. These are
exploratory because the validation split was already viewed during development.

## Acceptance gates

Keep a LoRA adapter only when all are true:

1. 100% schema-valid output after normalization.
2. Higher RETAKE recall or lower false-READY rate than frozen Gemma.
3. No patient leakage and no test-set tuning.
4. The system still abstains on unsupported or uncertain images.
5. Outputs remain strictly about image quality, never diagnosis.

## Evaluate the frozen or tuned model

The default launcher selects the tuned bundle when it is complete. Record that
identity in the output filename:

```bash
./ml/serve_local.sh

python3 ml/evaluate_local.py \
  --limit 20 \
  --sampling stratified \
  --seed 42 \
  --output ml/runs/tuned-q4km-lora-val-smoke.json
```

To measure the frozen QAT fallback instead, stop the tuned server and launch
the fallback directory explicitly on a separate port:

```bash
RETINA_READY_MODEL_DIR="$PWD/models/gemma-4-26b-q4" \
RETINA_READY_PORT=8082 \
./ml/serve_local.sh

python3 ml/evaluate_local.py \
  --base-url http://127.0.0.1:8082/v1 \
  --output ml/runs/frozen-qat-q4-val.json \
  --progress-every 10
```

The evaluator reports schema validity, a READY/RETAKE confusion matrix,
RETAKE recall, false-READY rate, accuracy, and median/p95 latency. Invalid
responses and `LIMITED` abstentions remain in the headline metric denominator;
    they are never silently dropped. The test split has already been opened once
    for the frozen smoke adapter and must not be reused for iteration.

On the A100, use the same pinned HF checkpoint and evaluator for the fair
base-versus-adapter comparison. Dry-run validates the exact rows without
loading weights:

```bash
python3 ml/evaluate_peft.py \
  --dry-run --limit 20 --sampling stratified --seed 42
```

For a reproducible HF base-versus-selected-adapter audit, run both with
identical selection arguments and separate output files:

```bash
python3 ml/evaluate_peft.py \
  --limit 20 --sampling stratified --seed 42 \
  --output ml/runs/frozen-hf-val-smoke.json

python3 ml/evaluate_peft.py \
  --adapter-dir ml/runs/gemma4-26b-retina-decision-smoke \
  --limit 20 --sampling stratified --seed 42 \
  --output ml/runs/tuned-hf-val-smoke.json
```

Adapter evaluation requires the trainer's completed `run_provenance.json` and
verifies that all model and processor IDs and revisions exactly match the
frozen comparison source before any weights are loaded.

For an adapter trained with `loss_scope=decision_token`, audit the exact
one-token classifier without autoregressive JSON generation:

```bash
python3 ml/evaluate_decision_logits.py \
  --adapter-dir ml/runs/gemma4-26b-retina-decision-smoke \
  --sampling stratified --seed 42 \
  --ready-threshold 0.988 \
  --output ml/runs/eval-decision-logits.json
```

The evaluator reconstructs the exact supervised prefix, proves that READY and
RETAKE share that prefix so the truth cannot leak into the score, truncates all
Gemma 4 sequence masks immediately before the decision token, and compares the
pinned READY/RE class logits. It reports per-class recall, false-READY rate,
balanced accuracy, and tie-aware ROC-AUC. Calibrate `--ready-threshold` on
validation only, freeze it, and then open the test split once.

## Recorded A100 result (2026-07-31)

- Hardware: one A100 SXM4 80 GB; peak allocated/reserved 49.49/49.63 GiB.
- Trainable adapter: 5,591,040 q/v LoRA parameters; frozen Gemma 4 MoE experts
  remained BF16 because their raw 3-D tensors are not bitsandbytes linears.
- Corrected smoke: 128 images, exactly 64 READY / 64 RETAKE, 60 optimizer
  steps, one supervised decision token per image.
- Strict generated JSON on 20 validation images: 80% accuracy, 100% schema
  validity, 100% RETAKE recall, 0 false READY, and 5/9 READY recall. The frozen
  HF base scored 60% and recognized 1/9 READY images on the identical rows.
- All 400 validation images at the naive 0.5 logit threshold: 68.5% accuracy,
  68.88% balanced accuracy, and ROC-AUC 0.750. This threshold was too
  permissive for safety.
- Frozen validation-calibrated threshold: READY only when `P(READY) > 0.988`.
- One-time 400-image sealed test: 64.25% accuracy, 61.09% balanced accuracy,
  29.44% READY recall, 92.73% RETAKE recall, 7.27% false READY, and ROC-AUC
  0.756.

The generated-output smoke is encouraging, while the full test result shows
that the adapter is not ready for clinical use. Keep the product framed as
capture-quality decision support with human review and fail-closed routing.

## Primary sources

- [Google Gemma 4 overview and memory table](https://ai.google.dev/gemma/docs/core)
- [Official Gemma 4 26B A4B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
- [Official Gemma 4 26B A4B base checkpoint](https://huggingface.co/google/gemma-4-26B-A4B)
- [Official Gemma 4 26B A4B instruction checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [Google vision QLoRA guide](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora)
- [Google Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
