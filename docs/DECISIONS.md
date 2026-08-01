# Implementation decisions

## Scope

RetinaReady assesses the **technical capture quality** of conventional color
fundus photographs. It does not assess OCT or angiography, and it does not
diagnose or rule out disease.

The public API returns four states:

- `READY`: technically suitable for downstream review.
- `LIMITED`: potentially usable, but quality is borderline or uncertain.
- `RETAKE`: insufficient technical quality; one capture action is supplied.
- `UNSUPPORTED`: the input is not a supported color fundus photograph.

## Data

DeepDRiD is the primary dataset because it includes overall image-quality,
artifact, clarity, and field-definition labels. Splits must remain patient
separated. Raw images and generated training artifacts stay outside version
control.

AngioReport is not a training source: it is angiography, does not provide the
required quality labels, excludes many unusable images, and contains many
correlated frames per eye.

## Model deployment

The primary offline demo uses a fresh Q4_0 conversion of the exact standard
Gemma 4 26B-A4B base used for training, its matching BF16 projector, and the
selected RetinaReady adapter as a separate F32 LoRA loaded by `llama.cpp`.
An exact-base Q4_K_M alternate is retained for compatibility experiments, but
it is not accepted for full-Metal vision inference on the tested 24-GB Mac.
Keeping the LoRA separate avoids losing small learned updates during a BF16
merge and subsequent quantization. Google's separately trained QAT Q4_0 GGUF
remains an untuned fallback and is never combined with this adapter.

The local server is bound to `127.0.0.1`, uses one slot and a bounded context,
and receives image bytes only as an in-memory request. The Q4 GGUFs are
inference artifacts, not trainable checkpoints. The A100 trainer used a
partially quantized PEFT path: supported 2-D attention linears were NF4, raw
3-D MoE experts remained frozen BF16, and 5,591,040 rank-16 q/v LoRA parameters
trained. The 24-GB M5 Pro is suitable for tightly configured local inference
but not for 26B multimodal training.

The accepted local profile is exact-base Q4_0 with GPU projector offload and
the F32 LoRA. It completed 4/4 schema-valid requests in a four-image validation
smoke, with 75% accuracy, 100% RETAKE recall, zero false READY calls, and about
three-second median latency. That small run is an integration gate, not a
clinical estimate. Q4_K_M loaded under two clean-memory full-Metal profiles but
failed on the first vision request with a Metal out-of-memory error; its
183.51-second CPU-projector success is compatibility evidence only.

## Evaluation gate

The adapter is kept only if it improves the frozen baseline on a patient-held
out validation split. Primary metrics are:

1. REJECT/RETAKE recall.
2. False-ready rate.
3. Structured-output validity.
4. Abstention behavior on unsupported inputs.
5. Median and p95 local latency.

No headline metric may be computed from an image-level split that leaks a
patient across partitions.

The validation-frozen `P(READY) > 0.988` result belongs to the HF PEFT
direct-logit evaluator. The accepted llama.cpp runtime generates JSON and does
not enforce that threshold, so local GGUF smoke results are always reported
separately and malformed or uncertain output fails closed to `LIMITED`.
