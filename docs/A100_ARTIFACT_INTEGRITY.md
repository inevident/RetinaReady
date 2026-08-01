# A100 artifact collection and integrity

`scripts/collect_a100_artifacts.py` provides the non-destructive handoff from
the A100 host. It mirrors only an explicit allowlist into
`outputs/a100-retinapriority-20260801/mirror/` and never deletes remote or local
files.

The `finished-v1` profile includes the completed smoke and full-profile root
adapters, their provenance and JSON evaluation artifacts, the explicitly
evaluated full-profile checkpoint 74 adapter, escalation training configs,
RetinaPriority logs, and LoRA-named GGUF files. It excludes the still-mutating
challenger. Once that run's `run_provenance.json` has `status: completed`, use
`with-challenger-v1` to append it:

```sh
python3 scripts/collect_a100_artifacts.py --profile finished-v1
python3 scripts/collect_a100_artifacts.py --profile with-challenger-v1
```

The challenger profile fails closed while its provenance status is anything
other than `completed`. A read-only recheck is available after collection:

```sh
python3 scripts/collect_a100_artifacts.py \
  --profile with-challenger-v1 \
  --verify-only
```

The separate `cross-run-posttrain-v1` profile captures the complete immutable
evidence namespace at
`ml/runs/gemma4-26b-retinapriority-cross-run-posttrain-v1/` together with its
exact `retinapriority-cross-run-posttrain-v1.log`. It refuses to read the tree
until `posttrain-completion.json` records
`completed_research_evaluation_not_runtime_promotion` and its canonical
integrity binding verifies. It does not pull any adapters, general logs,
configs, or GGUFs from the standard profiles:

```sh
python3 scripts/collect_a100_artifacts.py --profile cross-run-posttrain-v1
python3 scripts/collect_a100_artifacts.py \
  --profile cross-run-posttrain-v1 \
  --verify-only
```

Once the first successful cross-run manifest is recorded, subsequent
collections and verifications must match it exactly. The collector will not
replace that manifest with a changed remote tree or log; a new post-training
attempt requires a new versioned profile and evidence namespace.

The completed `cross-run-posttrain-v1` collection contains 19 files totaling
1,030,753 bytes. Its remote and locally recomputed manifests are byte-identical
with SHA-256
`89a278d01bd534b4d584d61ed3d8bbdbe431d2c23d82af20e5bc8c6b966cf7da`.
The comparison reports no missing, unexpected, hash-mismatched, or
size-mismatched files, and a second `--verify-only` pass matched the immutable
record. The evidence is under
`outputs/a100-retinapriority-20260801/mirror/ml/runs/gemma4-26b-retinapriority-cross-run-posttrain-v1/`;
the machine-readable comparison is
`outputs/a100-retinapriority-20260801/integrity/cross-run-posttrain-v1.comparison.json`.

For each profile the collector writes a remote manifest, a locally recomputed
manifest, and a comparison record under `outputs/.../integrity/`. Manifests
contain sorted repository-relative paths, byte sizes, and SHA-256 hashes; no
timestamps or host-dependent absolute paths are included. A successful run
requires byte-identical local and remote manifests. It also requires the
remote manifest to remain unchanged across the transfer, retrying a bounded
number of times if an allowlisted file changes while being copied.

The allowlist intentionally excludes:

- Gemma base-model and multimodal-projector weights;
- Hugging Face caches and credentials;
- optimizer, scheduler, RNG, and general checkpoint state;
- TensorBoard event files;
- duplicated 31 MB tokenizer payloads (the small tokenizer/config bindings and
  base model remain sufficient for this LoRA handoff);
- symlinks, arbitrary logs, and any single file larger than 256 MiB.

For the post-training evidence tree, the same limits also exclude cache,
checkpoint, TensorBoard, optimizer, scheduler, RNG, tokenizer-payload, adapter,
and GGUF files while retaining every other regular evidence file recursively.

Do not terminate or destroy the A100 instance until the final winning profile
reports `"matched": true` and the verified artifacts have passed local QA.
