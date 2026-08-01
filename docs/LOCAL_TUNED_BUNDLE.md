# Tuned local Gemma 4 bundle

RetinaReady's tuned local model is a llama.cpp bundle with two quantizations of
the exact same base. The smaller Q4_0 base is the accepted 24-GB Mac profile;
Q4_K_M is retained as a higher-bit compatibility artifact, not as an
interchangeable live-demo profile. One base, the projector, and the LoRA are
required for inference, and deployed inference remains on the Mac:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `retinaready-gemma4-26b-a4b-q4_0.gguf` | 14,439,363,136 | `41b9cbf3896a518a3fc8bd8b70fcc05fe4735a2474783c0d4df3a8c32975b5bb` |
| `retinaready-gemma4-26b-a4b-q4_k_m.gguf` | 16,796,017,216 | `64f4edd63a5f171912075726c3045b9c6a7283d1595f0fcc7fbd356862487879` |
| `retinaready-gemma4-26b-a4b-mmproj-bf16.gguf` | 1,194,827,808 | `2413217255d10cf9fc13a2756b448e4760f2fc945cfec2d2b6100a0f74b39ca7` |
| `retinaready-gemma4-26b-a4b-retina-decision-lora-f32.gguf` | 22,372,352 | `e20c573c8ca5cf40d0027a92285f055b18eeae73c3b7088a81f0d6c7853bbf62` |

The source checkpoint and processor are pinned independently:

- Base: `google/gemma-4-26B-A4B-it` at
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- Processor/chat template: `google/gemma-4-E2B-it` at
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- Selected PEFT adapter SHA-256:
  `f3739472b5abd66e93a026de9bf87c63cb3b34e28f46452275e9f99c11bb18be`
- Converter and remote-validation source: llama.cpp tag `b10180`, commit
  `11b068d06605288ce7917534b46d52b47823dc13`
- Local acceptance runtime: Homebrew llama.cpp build `10210`, commit
  `000547513`

## Why the LoRA stays separate

Do not merge this adapter into BF16 or attach it to Google's existing QAT
Q4_0 model. It was trained against the standard BF16 checkpoint, while the
QAT model is a different checkpoint. A controlled merge audit also found
that BF16 addition rounded roughly 44-56% of individual LoRA delta entries to
zero and caused substantial last-token-logit drift. Baking that result into
Q4 would discard still more of the trained update.

The faithful llama.cpp layout quantizes the exact matching base, keeps the
small adapter in F32, and applies it at runtime after base-weight
dequantization:

```bash
llama-server \
  --model models/retinaready-gemma4-26b-tuned/retinaready-gemma4-26b-a4b-q4_0.gguf \
  --mmproj models/retinaready-gemma4-26b-tuned/retinaready-gemma4-26b-a4b-mmproj-bf16.gguf \
  --lora models/retinaready-gemma4-26b-tuned/retinaready-gemma4-26b-a4b-retina-decision-lora-f32.gguf \
  --alias retinaready-gemma4-26b \
  --host 127.0.0.1 \
  --ctx-size 2048 \
  --batch-size 512 \
  --ubatch-size 512 \
  --mtmd-batch-max-tokens 512 \
  --parallel 1 \
  --n-gpu-layers 999 \
  --reasoning off \
  --reasoning-budget 0 \
  --jinja
```

This is a tuned local model even though it is packaged as a base, projector,
and adapter rather than one lossy merged file.

## Launch

From the project root:

```bash
./ml/serve_local.sh
```

The launcher selects this tuned directory only when the projector, LoRA, and
at least one exact-base quantization are present and non-empty. It prefers
Q4_0 on a 24-GB Mac and uses Q4_K_M when Q4_0 is absent. Otherwise it falls
back to the untuned official QAT bundle. The paths can be set explicitly when
needed:

```bash
RETINA_READY_MODEL_FILE=/absolute/path/base.gguf \
RETINA_READY_MMPROJ_FILE=/absolute/path/mmproj.gguf \
RETINA_READY_LORA_FILE=/absolute/path/adapter.gguf \
./ml/serve_local.sh
```

Keep the server bound to `127.0.0.1`. The default 2048-token, one-slot profile
uses 512-token language and multimodal batches because Gemma's non-causal
280-token image chunk requires a micro-batch of at least 280. Q4_0 enables GPU
projector offload by default. For the larger Q4_K_M alternate, the launcher
disables projector offload unless `RETINA_READY_MMPROJ_OFFLOAD=on` is set; CPU
projector mode fits but is a slow compatibility path. Do not use Q4_K_M as the
live-demo profile on this Mac: both clean-memory full-Metal attempts loaded the
model but failed on the first vision request with a Metal out-of-memory error.
`RETINA_READY_GPU_LAYERS` controls partial offload for diagnostic experiments.

## Conversion record

The standard BF16 base and matching projector were exported on the A100 host
with the pinned llama.cpp converter. The base dry run mapped 658 tensors and
the projector dry run mapped 356 tensors. The adapter converter mapped all
110 LoRA tensors and wrote them as F32. The 50.5-GB temporary BF16 text GGUF
was quantized twice. The 14.44-GB Q4_0 file is 4.57 bits per weight and is the
24-GB Mac default. The 16.80-GB Q4_K_M file is 5.32 bits per weight; 60 tensors
whose dimensions do not meet K-quant block requirements used llama.cpp's
documented fallback types. Neither filename means every tensor is literally
four bits.

Remote validation loaded the Q4 base, applied the complete LoRA, loaded the
matching multimodal projector, reached a healthy llama.cpp server, and
returned schema-valid non-diagnostic JSON for real DeepDRiD validation images.
On the Mac, Q4_0 with GPU projector offload passed the final acceptance gate.
The latest clean retry loaded in about 4.54 seconds (an earlier retry took about
7.66 seconds). Prior READY and RETAKE smokes completed in about 3.44 and 3.18
seconds. A four-image validation smoke completed 4/4 requests with 4/4 valid
schemas, 75% accuracy, 100% RETAKE recall, zero false READY calls, 3035.705-ms
median latency, and 3281.666-ms p95 latency. A real API request completed in
about 2.2 seconds. The configured `retinaready-gemma4-26b` alias and loaded-LoRA
endpoint were both verified.

Q4_K_M is not accepted for full-Metal multimodal inference on this 24-GB Mac.
With clean memory, it loaded in about 8.10 seconds under the default profile and
3.68 seconds at context 1024 / batch 128, but the first vision request failed
both times with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Disabling
projector offload completed one schema-valid request in 183.51 seconds, which
is retained only as a slow CPU-projector compatibility proof. The complete
machine-readable record is `outputs/local-runtime-validation-20260731.json`.

## Safety and evaluation

This model grades acquisition quality only. `READY` means suitable for human
review; it never means the eye is healthy. In the **HF PEFT direct-logit
evaluator**, the sealed-test result at the validation-frozen
`P(READY) > 0.988` policy was 92.73% RETAKE recall and 7.27% false READY, so
this is a hackathon research prototype rather than a clinical device. The
current llama.cpp path performs free-form JSON generation and does not enforce
that direct-logit threshold. Free generation is therefore a separate runtime
smoke result, not a claim that the calibrated HF policy carries over exactly.
The four-image local result is an integration smoke, not a clinical or
statistically meaningful performance estimate.

The app fails closed around the model. Uploaded bytes are held in memory, are
not written to disk, and are sent only to the loopback server. Unsupported or
uncertain images, contradictory output, and malformed model JSON normalize to
`LIMITED`; they cannot become `READY` through fallback parsing. If the large
model is not running, use the explicitly labelled deterministic presentation
mode rather than implying that an uploaded image received tuned inference.
