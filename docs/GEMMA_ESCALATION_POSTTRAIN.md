# Gemma Escalation Post-Training Orchestration

`ml/orchestrate_escalation_posttrain.py` is the fail-closed bridge from a
completed RetinaPriority training run to retrospective research evidence. It
does not promote a model into the application or authorize clinical use.

## Order of operations

1. Verify the completed full-run provenance and all quality-pass train/val
   manifest bindings.
2. Discover the completed run root plus each retained `checkpoint-N` epoch
   checkpoint.
3. Hash each adapter. Exact weight aliases are scored once; identical weights
   paired with different adapter configuration fail closed.
4. Run decision-token logits over every row of the quality-pass validation
   manifest. Existing reports are reused only after their manifest, split,
   coverage, adapter, provenance, logits, probabilities, predictions, and
   summary metrics are recomputed and verified.
5. Rank every unique adapter using validation only: highest ROC-AUC, fewest
   false `ROUTINE` decisions / highest `PRIORITY` recall, then highest balanced
   accuracy. The winner must pass every safety floor; a lower-ranked adapter is
   never silently substituted. Calibration and evaluation are not read or
   scored until the hash-bound selection report is durable.
6. Score the complete calibration and evaluation manifests with the frozen
   adapter, then invoke `calibrate_escalation_adapter.py`.

Additional completed adapters from other runs can be included with repeatable
`--candidate-dir` arguments. Their own provenance, source revisions,
decision-token objective, token IDs, adapter configuration, and weights are
verified, but their manifest paths are ignored. Every candidate is rescored on
the one validation manifest locked by `--config`; the same config exclusively
owns the downstream calibration and evaluation manifests.

## Research risk profile

The orchestration CLI exposes `--false-routine-risk`,
`--false-priority-risk`, and `--delta`. Defaults are `0.10`, `0.10`, and
`0.05`.

This is a finite-sample research default, not a clinical safety claim. With
35 `PRIORITY` calibration patients and 40 `ROUTINE` calibration patients, even
zero observed adverse events gives one-sided 95% Clopper-Pearson upper bounds
of 8.20% and 7.22%. Those denominators cannot certify a 5% risk limit.

## Run on the training host

```bash
cd /workspace/retina-ready
source .venv-a100/bin/activate
python3 ml/orchestrate_escalation_posttrain.py \
  --config ml/configs/gemma4_26b_escalation_quality_pass_full.json \
  --candidate-dir ml/runs/gemma4-26b-retinapriority-decision-smoke \
  --candidate-dir ml/runs/gemma4-26b-retinapriority-quality-pass-qv-challenger-v1 \
  --work-dir ml/runs/gemma4-26b-retinapriority-cross-run-posttrain-v1 \
  2>&1 | tee ml/runs/logs/retinapriority-cross-run-posttrain-v1.log
```

The utility writes a provenance-rich `checkpoint-selection.json`, complete
decision-logit reports, `selective-policy-evaluation.json`, and a final
`posttrain-completion.json` beneath `--work-dir` (or the primary run's
`posttrain/` directory when omitted). If an existing report is partial, stale,
corrupted, or bound to different risk parameters, the command stops rather
than overwriting or reusing it.

The primary config automatically contributes its completed run root and every
valid retained epoch checkpoint beneath that root, so do not repeat that run
with `--candidate-dir`. Each explicit candidate path must point directly to a
completed adapter directory containing non-symlink `adapter_config.json`,
`adapter_model.safetensors`, and `run_provenance.json` files. The safetensors
bundle is materialized on CPU and checked against its LoRA configuration and
recorded trainable-parameter inventory before any cached report can be reused.
The candidate's own train/validation/calibration/evaluation manifest names are
ignored; all candidates use the manifests owned by the primary config.

If the q/v challenger has not yet recorded `status: completed` with no failure,
the command intentionally stops. Work directories are immutable evidence
namespaces. After changing orchestration code or any candidate, use a new
versioned `--work-dir` (or deliberately archive the previous evidence); do not
delete or overwrite evidence automatically.
