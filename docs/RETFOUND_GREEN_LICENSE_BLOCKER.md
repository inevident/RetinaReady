# RETFound-Green challenger: licence blocker

## Outcome

The proposed RETFound-Green frozen-encoder escalation challenger was stopped
before weights were downloaded or training began. It is not integrated into
the application, workflow, or UI.

This is an engineering compatibility decision, not legal advice. The project
uses the stricter reading because the upstream licence terminates rights on a
breach and the hackathon requires public distribution.

## Why it is blocked

RETFound-Green is scientifically attractive: it is a compact
`vit_small_patch14_reg4_dinov2` retinal encoder operating at 392 by 392 pixels,
and the authors report strong linear-probe results for grade-2-or-worse
diabetic-retinopathy tasks. The official instructions publish a v0.1
state-dict release and specify three-channel mean and standard deviation 0.5.

The current repository is licensed under **Justin's Custom Non-Commercial
Research Licence (CNCRL)**. The licence allows non-commercial research but
withdraws that permission from an `Industry-Involved Project`. Its definition
includes a project where a commercial entity provides funding, resources, or
personnel, and it requires prior written permission for such use.

The official event page says that Build with Gemma NYC is approved and
prize-funded by Google's Gemma team, offers cash prizes, judges submissions,
and requires a public repository and Kaggle writeup. That satisfies the
licence's commercial-entity involvement test even if the team itself is not a
company. No written permission from the RETFound-Green licensor is present.

Primary sources:

- [Pinned RETFound-Green licence](https://github.com/justinengelmann/RETFound_Green/blob/767c77ecc6ad2656ace051b17bf22d2b47485c6c/LICENSE)
- [Official RETFound-Green repository and loading instructions](https://github.com/justinengelmann/RETFound_Green/tree/767c77ecc6ad2656ace051b17bf22d2b47485c6c)
- [RETFound-Green paper](https://www.nature.com/articles/s41467-025-62123-z)
- [Official hackathon page](https://gemmanycaihealthcare.devpost.com/)

## Reproduce the audit

Run:

```bash
python3 ml/audit_retfound_green_license.py
```

The script fetches only the pinned upstream `LICENSE`, verifies its SHA-256 and
the relevant clauses, verifies the exact escalation manifests and frozen
DenseNet control report, writes
`outputs/retfound-green-escalation/license-blocker.json`, and exits `2` for the
expected licence block. It contains no weight-download or training code.

Pinned input hashes:

| Input | SHA-256 |
| --- | --- |
| `train.csv` | `3edb9fe596ffd61439caacaf6ddf9cd9299ca4e3a689b26f020c5674e33d3992` |
| `val.csv` | `f24f8375c6b4c6925e4049b862ef56d7cf75bcada116d957af673e0f048920d5` |
| `calibration.csv` | `ce81aba4321b0e16e2ac58bdf0d6b37eedfca3bdf4b86e97b71271f01fcb363a` |
| `eval.csv` | `796b2afdcfdb84b431ea0d0d3f4329d08b183720cb038f555fb7db3fe09ae1b4` |
| DenseNet control report | `3d3798692ec04649d079de3b1d2045bcc79f5ba732557d70b6decb0aa8e9d984` |

The frozen control's official-validation results remain the only disease
priority result in this branch: image ROC-AUC 0.9300, patient ROC-AUC 0.9516,
patient aggregate coverage 37%, accepted accuracy 97.30%, no false ROUTINE
patient decision among 50 priority-truth patients, and one false PRIORITY
patient decision among 50 routine-truth patients. These are exploratory, not
clinical claims.

## Unblock condition

Obtain prior written permission directly from the RETFound-Green licensor that
explicitly covers this Google-funded public hackathon submission. Preserve the
permission artifact locally, record its path and SHA-256 in
`ml/configs/retfound_green_license_audit.json`, and rerun the audit. Only after
it returns `READY_FOR_EXPERIMENT_SETUP` should anyone download the 87.4-MB
state dict or implement the frozen-head experiment.

If permission cannot be obtained, use a foundation model whose licence
expressly permits this event context. Any alternative must undergo its own
licence and pretraining-overlap audit before comparison on DeepDRiD.
