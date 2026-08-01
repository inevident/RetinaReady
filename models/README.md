# Model artifact policy

This directory deliberately separates compact, project-specific runtime
artifacts from multi-gigabyte base weights and experimental outputs.

## Versioned in Git

- `retinaready-quality-specialist/`: the frozen DenseNet-121 backbone,
  decision/factor heads, and policy manifest used by the first-stage technical
  quality gate (about 35 MB total).
- `retinapriority-gemma4-26b/`: the selected F32 llama.cpp LoRA, checksum,
  model card, and evaluation manifest used by the second-stage review-priority
  route (about 22 MB total).
- `retinaready-escalation-demo/promotion-manifest.json`: the frozen promotion
  record consumed by the demo harness.

These files are below GitHub's per-file Git limit and are required to inspect
or run the project-specific parts of the workflow.

## Kept outside Git

The full local Gemma bundle is excluded because its individual files are far
larger than ordinary Git hosting permits:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `retinaready-gemma4-26b-a4b-q4_0.gguf` | 14,439,363,136 | `41b9cbf3896a518a3fc8bd8b70fcc05fe4735a2474783c0d4df3a8c32975b5bb` |
| `retinaready-gemma4-26b-a4b-q4_k_m.gguf` | 16,796,017,216 | `64f4edd63a5f171912075726c3045b9c6a7283d1595f0fcc7fbd356862487879` |
| `retinaready-gemma4-26b-a4b-mmproj-bf16.gguf` | 1,194,827,808 | `2413217255d10cf9fc13a2756b448e4760f2fc945cfec2d2b6100a0f74b39ca7` |

The accepted Mac profile uses the Q4_0 base plus the projector. Q4_K_M is a
retained compatibility artifact and is not the tested live-demo profile on a
24-GB Mac. Do not substitute Google's separate QAT Q4_0 checkpoint: the LoRA
was trained against the standard Gemma 4 26B-A4B checkpoint and the two bases
are not interchangeable.

See `docs/LOCAL_TUNED_BUNDLE.md` and `docs/A100_HANDOFF.md` for exact source
revisions, conversion steps, launch variables, checksums, and validation
boundaries. Raw datasets, training checkpoints, caches, and evaluation outputs
are likewise reproducible local inputs/outputs and remain outside Git.

## Hosting recommendation

Publish the large base/projector as a versioned Kaggle Model (or another model
registry that accepts files of this size), not as Git objects. Keep the public
GitHub repository as the source of truth for code, compact adapters, manifests,
checksums, and reproduction instructions. Do not duplicate Google's base
weights unless their license and the chosen registry's terms allow it.
