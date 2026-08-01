# RetinaReady + RetinaPriority workflow contract

This document describes the product shell and safety boundary. It does not
claim that a clinical review-priority model has been validated or deployed.

The accepted demo path is `COMBINED`, launched with
`./scripts/run_priority_demo.sh`: the compact RetinaReady quality specialist
runs first, and only an exact `READY` result reaches the validation-selected
q/v rank-16 step-60 Gemma RetinaPriority LoRA. The live llama.cpp path is
pinned to the fixed demo inputs and is explicitly uncalibrated free generation.

## Modes

- `QUALITY_ONLY` runs the unchanged RetinaReady quality analyzer and returns
  `READY`, `RETAKE`, or `LIMITED`.
- `ESCALATION_ONLY` calls the review-priority adapter without a quality gate.
  This is a test surface for evaluating that adapter independently, not the
  final demo workflow.
- `COMBINED` runs quality first. Only an exact `READY` result is passed to the
  review-priority adapter. `RETAKE`, `LIMITED`, `UNSUPPORTED`, an exception, or
  malformed quality output blocks that stage.

The existing `POST /api/analyze` quality-only contract remains unchanged. The
new endpoint is `POST /api/workflow`; the mode is selected with the
`X-Product-Mode` header.

## Quality analyzer profiles

Product mode and quality-analyzer profile are independent controls. Historical
quality-only profiles can still use the earlier Gemma confirmation hybrid or
the deterministic presentation engine, but the final combined launcher fixes
the quality stage to the frozen 33 MB compact specialist. Gemma is reserved for
the downstream review-priority stage.

With **Combined** selected through `./scripts/run_priority_demo.sh`, the
exact-hash RetinaReady specialist executes in-process first. Only `READY`
reaches the separately served Gemma escalation LoRA over loopback; `LIMITED`
and `RETAKE` never call Gemma. The quality runtime checks independent
code-pinned hashes for its manifest, DenseNet backbone, and two heads before
every decision.

Because the compact quality model has no separately validated modality/OOD
gate, this profile is intentionally dataset-demo-only: it accepts exactly the
four pinned DeepDRiD images served by the ROUTINE, READY, LIMITED, and RETAKE
sample API keys. The UI presents those buttons as **Routine**, **Priority**,
**Limited**, and **Retake**; Priority deliberately retains the existing READY
API key. Routine uses `146_l2` from the training-split calibration slice. It is
a calibration/demo example, not held-out evidence. Any other upload, including
another decodable image, returns `LIMITED` before model inference. A
missing/tampered bundle, dependency error, decode failure, inference error,
malformed assessment, `RETAKE`, or `LIMITED` blocks the priority stage. Do not
describe the quality stage itself as Gemma inference or as an arbitrary-upload
quality checker.

## Review-priority decisions

The typed priority schema has exactly three states:

- `ROUTINE_REVIEW`: the usable image received no model priority flag and stays
  in the routine clinician-review queue. This does **not** mean healthy, normal,
  or no disease.
- `PRIORITY_REVIEW`: the usable image is flagged for earlier clinician review
  because the model detected potentially concerning signal. This is not a
  diagnosis.
- `UNCERTAIN`: the model cannot safely assign review order. Route to human
  prioritization and do not delay review based on the model.

Only an adapter result that is executed, locally available, structurally valid,
and explicitly release-enabled can release `ROUTINE_REVIEW` or
`PRIORITY_REVIEW`. Any missing artifact, exception, malformed output, or failed
release check is converted to `UNCERTAIN`. All displayed queue guidance is
policy-authored rather than model-authored.

## Historical compact-priority fallback

`build_escalation_adapter()` supports two separately selectable nonclinical
research-demo adapters. The raw app default retains
`RETINA_ESCALATION_ENGINE=specialist` for backward compatibility, but this
compact adapter is not the final selected RetinaPriority path. It is disabled
unless the common explicit opt-in is set and may be used only to reproduce the
earlier low-memory presentation fallback:

```bash
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_DEVICE=cpu
```

The promotion root is
`models/retinaready-escalation-demo/promotion-manifest.json`. It binds these
exact files:

- experimental specialist SHA-256
  `d01df16b339b4e262af67e57e2e32c14144047e970599621ca302f2d48314f84`;
- evaluation report SHA-256
  `3d3798692ec04649d079de3b1d2045bcc79f5ba732557d70b6decb0aa8e9d984`;
- DenseNet-121 backbone SHA-256
  `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`.

This external allowlist does not modify or contradict the source artifact. Its
`experimental_only=true`, `runtime_integration_authorized=false`, and
`diagnostic_use_authorized=false` flags remain intact. The manifest's authority
is narrowly scoped to an opted-in, local, nonclinical research demonstration.

At initialization and before every release attempt, the adapter re-hashes the
artifact, report, and backbone and verifies their schemas, source flags,
architecture, feature normalization, ensemble width, and exact strict
thresholds. It uses the baseline black-border crop, square padding, 512-pixel
resize, `[-1,1]` normalization, frozen DenseNet global features, five MLP
heads, and stored feature mean/std. Internal `ROUTINE` and `PRIORITY` map to
`ROUTINE_REVIEW` and `PRIORITY_REVIEW`; threshold equality and the entire gap
map to `UNCERTAIN`.

Missing opt-in, revoked opt-in, checksum or report drift, missing dependency,
decode failure, unsupported content type, non-finite score, or inference error
fails closed to `UNCERTAIN` with `release_allowed=false`. No network or cloud
resource is used.

The control evaluation is documented in `docs/ESCALATION_BASELINE.md`. Its
100-patient official-validation result is useful hackathon evidence, not
clinical validation; the documented recommendation remains do not integrate
for clinical use.

## Final selected Gemma LoRA path

The accepted launcher sets `RETINA_ESCALATION_ENGINE=gemma` and starts or
verifies a separate llama.cpp server carrying the selected RetinaPriority LoRA:

```bash
./scripts/run_priority_demo.sh
```

That launcher is the accepted path: it checksum-pins the converted LoRA and
verifies the exact model alias, absolute adapter path, and scale 1. The manual
environment contract is:

```bash
export RETINA_ANALYZER=specialist
export RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1
export RETINA_ESCALATION_ENGINE=gemma
export RETINA_ESCALATION_GEMMA_API_URL=http://127.0.0.1:8082
export RETINA_ESCALATION_GEMMA_MODEL_ID=retinapriority-gemma4-26b
export RETINA_ESCALATION_GEMMA_LORA_PATH=/absolute/path/to/retinapriority-lora-f32.gguf
export RETINA_ESCALATION_GEMMA_LORA_SHA256=<exact-lowercase-sha256>
```

This arrangement loads no Gemma quality model: the 33 MB RetinaReady
specialist runs first, and only an exact `READY` result reaches the one 26B
server carrying the escalation LoRA. The adapter permits only uncredentialed
HTTP loopback URLs. Before every inference it verifies the local LoRA SHA-256,
`/health`, the exact model alias in `/v1/models`, and exactly one active adapter
whose `/lora-adapters` path equals the configured resolved absolute path and
whose scale is 1.

The adapter's system prompt, user prompt, schema-constrained field names,
disclaimer, and internal
`ROUTINE` / `PRIORITY` labels match `ml/train_qlora.py`. Application parsing is
deliberately stricter than the general quality adapter: markdown fences,
reasoning prefixes, extra fields, non-null confidence, altered policy prose,
or any label other than `ROUTINE` / `PRIORITY` fail to `UNCERTAIN`. Only the
exact SHA-256 values of two quality-passing DeepDRiD demo inputs are accepted:
the training-split `146_l2` ROUTINE calibration/demo example and the existing
READY-keyed `296_l2` PRIORITY validation example. The routine example is not
held-out evidence. The fixed RETAKE and LIMITED examples remain blocked by
quality and are also rejected by the priority adapter itself.

This adapter is labelled
`gemma-lora-free-generation-uncalibrated-experimental`. Hugging Face
direct-logit probabilities and calibrated thresholds do not transfer to
llama.cpp free generation, so the runtime returns no confidence and makes no
clinical-performance claim. A successful token decision is usable only for
the explicitly opted-in hackathon research presentation.

The selected artifact is the q/v rank-16 step-60 adapter chosen on the frozen
70-image validation manifest. In its separate 182-image direct-logit
evaluation, it achieved priority-positive ROC-AUC 0.956903 and balanced
accuracy 0.912355 at threshold 0.5. The calibration-frozen selective policy
accepted 65/182 images, and all 65 accepted routes were correct. These figures
belong only to the offline Hugging Face direct-logit evaluator; the pinned live
llama.cpp free-generation labels do not inherit its thresholds, coverage, or
confidence claims.

## Verification

`app/tests/test_workflow.py` exercises the product modes and both the final and
historical adapter boundaries:

- all three modes;
- `READY`, `RETAKE`, `LIMITED`, and `UNSUPPORTED` quality branches;
- released routine and priority review suggestions;
- missing, failing, malformed, and non-releaseable priority adapters;
- quality-stage failure;
- non-diagnostic, policy-authored routine wording;
- exact artifact/report/backbone hash binding and unchanged source flags;
- opt-in, revoked opt-in, post-load tampering, decode, and content-type failures;
- preprocessing equivalence and reproduction of frozen report decisions;
- a real decisive image through the combined quality-first workflow.

`app/tests/test_analyzer.py` separately exercises the specialist-only profile,
including local-only health metadata, strict score/threshold consistency,
per-request independently pinned bundle re-verification, fixed-dataset input
scope, and fail-closed bundle/decode/inference paths.

`app/tests/test_gemma_escalation.py` uses mocked loopback responses to verify
URL restrictions, health/model/LoRA identity, post-start hash tamper handling,
the exact training prompt and target schema, fixed-input scope, ROUTINE and
PRIORITY mapping, and fail-closed network/schema/identity behavior.

The API tests also verify the exact four-sample mapping and hashes, READY-key
backwards compatibility, sample-button order, all three browser modes, the
fail-closed stub, the combined gate, and rejection of unknown product modes.
