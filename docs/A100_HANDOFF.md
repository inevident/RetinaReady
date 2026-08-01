# A100 handoff runbook

This runbook was executed on 2026-07-31 against one A100 SXM4 80 GB. It remains
the reproducible procedure for a future host; the completed artifacts are in
`outputs/a100-handoff-20260731/`.

Recorded outcome:

- DeepDRiD checksum, image integrity, and patient separation passed.
- The dtype-preserving one-step probe and both 60-step smoke runs completed.
- Peak CUDA allocated/reserved memory was 49.49/49.63 GiB.
- The selected adapter has 5,591,040 q/v LoRA parameters. Gemma 4's raw 3-D
  MoE experts stayed frozen BF16, so this was partially quantized PEFT rather
  than full-model 4-bit QLoRA.
- The corrected 64/64 balanced decision-token run reached 80% strict-JSON
  accuracy on the 20-image comparison set, with 100% schema validity, 100%
  RETAKE recall, zero false READY, and 5/9 READY recall.
- After a validation-only threshold freeze at `P(READY) > 0.988`, the one-time
  400-image sealed test achieved 92.73% RETAKE recall, 7.27% false READY,
  29.44% READY recall, 64.25% accuracy, and ROC-AUC 0.756.

Those metrics support a hackathon research demo, not clinical deployment.
This workflow avoids asking the 24 GB Mac to train or sending its 15 GB local
GGUF inference model to the GPU host.

For a new run, execute training commands only after SSH access, exact GPU
memory, storage, Hugging Face access, and the local input dry run have all been
verified.

## 1. Create the remote directory

Choose one explicit project path and use it consistently. These examples use
an SSH alias named `gpu-a100` and `/home/ubuntu/retina-ready`:

```bash
ssh gpu-a100 'mkdir -p /home/ubuntu/retina-ready'
```

Prefer an SSH alias in `~/.ssh/config` for identity files, ports, bastions, and
host-key policy. Do not put private keys, access tokens, or passwords in this
repository.

## 2. Preview and sync from the Mac

The helper requires both the host and remote directory, defaults to rsync dry
run, never removes remote files, and always excludes:

- `models/` and all GGUF/Safetensors files;
- outputs, checkpoints, and `ml/runs/`;
- Python virtual environments, `node_modules`, and caches.

Preview code and the small prepared manifests:

```bash
cd retina-ready
./scripts/sync_to_a100.sh \
  --host gpu-a100 \
  --remote-dir /home/ubuntu/retina-ready
```

An rsync dry run still opens an SSH connection, but it writes no files. Review
the itemized list before adding `--execute`.

Choose exactly one data path:

```bash
# Recommended: send the verified 1.3 GiB archive, not the 15 GB local model.
./scripts/sync_to_a100.sh \
  --host gpu-a100 \
  --remote-dir /home/ubuntu/retina-ready \
  --include-archive \
  --execute

# Faster remote setup but ~2.6 GiB over SSH: archive + extracted raw tree.
./scripts/sync_to_a100.sh \
  --host gpu-a100 \
  --remote-dir /home/ubuntu/retina-ready \
  --include-raw \
  --execute
```

Omit both data flags to transfer only code and manifests, then download the
pinned Zenodo archive from the A100 host later.

## 3. Remote hardware and storage preflight

After SSH access is available:

```bash
ssh gpu-a100
cd /home/ubuntu/retina-ready

uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
df -h .
```

The intended target is Linux with CUDA, BF16 support, and an A100 80 GB. An
A100 40 GB is below this project's 48 GiB guard for 26B multimodal QLoRA; do
not bypass that guard. Keep at least 150 GB free for the Hugging Face cache,
environment, adapters, checkpoints, and logs.

## 4. Restore and validate DeepDRiD

If the archive was synced, the existing data script detects it, verifies the
published byte size and MD5, extracts it, and rebuilds the manifests without
downloading it again:

```bash
./scripts/download_deepdrid.sh
```

If the extracted raw tree was synced instead:

```bash
python3 scripts/prepare_deepdrid.py
```

If neither was synced, `./scripts/download_deepdrid.sh` downloads the pinned
Zenodo release, then performs the same checks. The expected result is 1,200
training, 400 validation, and 400 sealed test images, with zero cross-split
patient overlap. Keep `data/manifests/test.csv` sealed until the adapter and
decision policy are frozen.

The DeepDRiD images are CC BY-SA 4.0 research data. Keep the raw directory
private to the authorized host and preserve the attribution in `data/README.md`.

## 5. Bootstrap the remote Python environment

The bootstrap installs packages and runs the read-only preflight; it does not
load a model or train:

```bash
./ml/bootstrap_a100.sh
source .venv-a100/bin/activate
```

Verify CUDA, BF16, and the visible device:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.is_bf16_supported(), "GPU/runtime does not support BF16"
props = torch.cuda.get_device_properties(0)
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": props.name,
    "memory_gib": round(props.total_memory / 2**30, 1),
})
PY
```

Record the environment before any run:

```bash
mkdir -p ml/runs/environment
python -m pip freeze > ml/runs/environment/pip-freeze.txt
nvidia-smi > ml/runs/environment/nvidia-smi.txt
```

## 6. Verify pinned Hugging Face access

The future trainer pins these public repositories and revisions:

- `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`

They were public and ungated when this handoff was prepared. The preflight
checks the pinned metadata and small JSON files anonymously without downloading
weights. A read-only token is optional unless the access policy changes:

```bash
hf auth login
hf auth whoami
```

Use a read-only token. Do not paste it into a script, commit it, include it in
logs, or transfer the Hugging Face cache back to the Mac.

The following command is safe to run before training: it reads all train/val
rows, verifies that every image exists, checks patient separation, and prints
the resolved configuration. It does not import the training stack, contact
Hugging Face, load weights, or use the GPU:

```bash
python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json \
  --dry-run
```

## 7. VRAM probe and smoke training

Do not run these commands on a new host until access and the preflight above
are complete.

The first future run is one image and one optimizer step. Its purpose is only
to prove the complete load/backward/save path and record peak CUDA memory:

```bash
mkdir -p ml/runs/logs
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONUNBUFFERED=1 python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_vram_probe.json \
  2>&1 | tee "ml/runs/logs/${run_id}-26b-vram-probe.log"
```

Only after that succeeds is the bounded 128-image, 60-step smoke appropriate:

```bash
mkdir -p ml/runs/logs
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONUNBUFFERED=1 python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json \
  2>&1 | tee "ml/runs/logs/${run_id}-26b-smoke.log"
```

The first full-JSON smoke is useful as a pipeline check but collapsed to the
majority RETAKE decision under free generation. The corrected bounded run is:

```bash
PYTHONUNBUFFERED=1 python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_decision_smoke.json \
  2>&1 | tee "ml/runs/logs/${run_id}-26b-decision-smoke.log"
```

Run it inside `tmux` so an SSH disconnect does not terminate it. The trainer
logs every five steps, evaluates and saves every 20 steps, retains the newest
two checkpoints, and writes TensorBoard events below
`ml/runs/gemma4-26b-retina-smoke/`.

The trainer never resumes implicitly. To resume after an SSH or host
interruption, name one validated checkpoint explicitly:

```bash
PYTHONUNBUFFERED=1 python3 ml/train_qlora.py \
  --config ml/configs/gemma4_26b_smoke.json \
  --resume-from-checkpoint ml/runs/gemma4-26b-retina-smoke/checkpoint-40 \
  2>&1 | tee -a "ml/runs/logs/${run_id}-26b-smoke.log"
```

Every run atomically updates `run_provenance.json` with the pinned sources,
manifest and selected-row hashes, effective configuration, packages, GPU,
resume checkpoint, failure state, metrics, and peak CUDA memory.
The data collator fails closed unless the rendered generation prompt is an
exact token prefix, and computes loss only on the assistant target rather than
the system/user prompt.
Fresh runs also refuse a non-empty output directory. A resume must name a
validated checkpoint directly inside that same output directory, preventing an
old adapter or provenance record from being overwritten accidentally.

To inspect TensorBoard without exposing it publicly, run this on the remote:

```bash
tensorboard --logdir ml/runs --host 127.0.0.1 --port 6006
```

Then forward it from the Mac:

```bash
ssh -L 6006:127.0.0.1:6006 gpu-a100
```

Open <http://127.0.0.1:6006> locally.

## 8. Artifacts and download-back

Expected smoke artifacts are under:

```text
ml/runs/gemma4-26b-retina-smoke/
ml/runs/logs/
ml/runs/environment/
```

Before keeping an adapter, require a clean save, finite/stable loss, and
improvement over the frozen validation baseline. Do not open the sealed test
split during iteration.

Download only adapters, tokenizer/processor metadata, metrics, logs, and the
environment record. Do not download the base Hugging Face cache, raw images,
optimizer state, or the local GGUF:

```bash
mkdir -p outputs/a100-handoff
rsync -avh --progress \
  --exclude='checkpoint-*/' \
  --exclude='optimizer.pt' \
  --exclude='scheduler.pt' \
  --exclude='rng_state.pth' \
  gpu-a100:/home/ubuntu/retina-ready/ml/runs/gemma4-26b-retina-smoke/ \
  outputs/a100-handoff/gemma4-26b-retina-smoke/

rsync -avh \
  gpu-a100:/home/ubuntu/retina-ready/ml/runs/logs/ \
  outputs/a100-handoff/logs/

rsync -avh \
  gpu-a100:/home/ubuntu/retina-ready/ml/runs/environment/ \
  outputs/a100-handoff/environment/
```

Keep the original remote run intact until the downloaded artifacts have been
checksummed and the adapter can be loaded in a separate evaluation workflow.

Before opening the sealed test split, compare the frozen pinned HF checkpoint
and adapter on the exact same validation rows:

```bash
python3 ml/evaluate_peft.py \
  --limit 20 --sampling stratified --seed 42 \
  --output ml/runs/frozen-hf-val-smoke.json

python3 ml/evaluate_peft.py \
  --adapter-dir ml/runs/gemma4-26b-retina-smoke \
  --limit 20 --sampling stratified --seed 42 \
  --output ml/runs/tuned-hf-val-smoke.json
```

## 9. Executed selection and sealed evaluation

The completed run selected the 60-step decision-token adapter, copied it to
`outputs/a100-handoff-20260731/final-adapter/`, and recorded its source hash
and pinned model revisions there. The exact 20-row generated-JSON comparison,
full 400-image validation logits, and one-time sealed test are retained under
`outputs/a100-handoff-20260731/run-records/`.

The `P(READY) > 0.988` policy was chosen from validation only and frozen before
the test split was opened. Do not recalibrate it on test results. Those direct
HF logits are not automatically reproduced by llama.cpp free generation; the
local JSON path is a separate deployment smoke test.

## 10. Executed local export

The exact standard base used for LoRA training was exported to BF16 GGUF with
llama.cpp tag `b10180` at commit
`11b068d06605288ce7917534b46d52b47823dc13`. The matching pinned processor
provided the vision projector and chat/tokenizer metadata. The selected PEFT
adapter was converted separately to F32 GGUF; it was not merged into BF16.

Two base quantizations were produced from the same 50,505,136,704-byte BF16
GGUF:

| Quantization | Bytes | Bits/weight | Purpose |
| --- | ---: | ---: | --- |
| Q4_0 | 14,439,363,136 | 4.57 | Default 24-GB Mac live-demo path |
| Q4_K_M | 16,796,017,216 | 5.32 | Higher-bit alternate / CPU-projector compatibility path |

The local runtime must load one exact-base GGUF plus the matching BF16
projector and F32 LoRA. Never attach this LoRA to Google's separately trained
QAT GGUF. File sizes, SHA-256 values, local runtime settings, and acceptance
results are maintained in `models/retinaready-gemma4-26b-tuned/` and
`docs/LOCAL_TUNED_BUNDLE.md`.
