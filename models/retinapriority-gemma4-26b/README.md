# RetinaPriority local Gemma 4 LoRA

> **Modification notice:** this is a derived Gemma 4 artifact changed by
> RetinaReady contributors through q/v-only LoRA training on the public
> DeepDRiD dataset and conversion to F32 GGUF with llama.cpp. The unmodified
> base weights are not included. See `NOTICE`, the repository's
> `THIRD_PARTY_NOTICES.md`, and `licenses/GEMMA-4-APACHE-2.0.txt`.

This directory contains the validation-selected q/v-only, step-60
RetinaPriority LoRA converted to an F32 llama.cpp adapter. Its historical
training-run name, `gemma4-26b-retinapriority-decision-smoke`, is retained in
`manifest.json` for provenance. The adapter is loaded on top of the existing
exact-base Gemma 4 26B-A4B Q4_0 model and its matching multimodal projector;
those large shared files are not duplicated here.

Start the complete quality-first workflow from the repository root:

```bash
./scripts/run_priority_demo.sh
```

The launcher verifies the LoRA checksum, exact model alias, exact active LoRA
path, and scale before starting the app. The compact 33 MB quality specialist
runs first. Only a `READY` quality result can reach this escalation adapter.

The UI exposes exactly four pinned presentation buttons:

| Button | Pinned example | Role in the demo |
| --- | --- | --- |
| Routine | `146_l2` | Training-partition calibration/demo example; not held-out evidence |
| Priority | `296_l2` | Validation example |
| Limited | `265_l2` | Quality gate blocks Gemma escalation |
| Retake | `431_l2` | Quality gate blocks Gemma escalation |

The adapter was selected solely on the frozen 70-image validation manifest.
Its separate direct-logit evaluation covered 182 quality-passing images and
achieved ROC-AUC 0.956903. The frozen selective policy accepted 65/182 images
and routed all 65 correctly, abstaining on the rest. These are research results
from patient-disjoint development partitions, not clinical validation.

The live llama.cpp path is deliberately labeled uncalibrated: constrained
free-generation labels are not calibrated probabilities and do not inherit the
offline direct-logit thresholds. The live adapter accepts only the two pinned
quality-passing DeepDRiD presentation examples; the Limited and Retake examples
are stopped by the quality gate. It returns `UNCERTAIN` on any identity,
checksum, schema, or input-scope failure.

See `manifest.json` for exact hashes, candidate selection results, policy
thresholds, and limitations.
