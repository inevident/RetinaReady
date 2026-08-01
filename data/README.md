# RetinaReady data

This directory is configured for the **DeepDRiD v1.1** public release. Raw
archives and images are deliberately ignored by version control. Download,
verify, extract, and prepare the dataset with:

```bash
./scripts/download_deepdrid.sh
```

That command obtains the pinned Zenodo release, verifies the published byte
size and MD5 checksum, extracts it into `data/raw/deepdrid-v1.1/`, and produces
patient-disjoint manifests in `data/manifests/`.

## Source and integrity

- Dataset: DeepDRiD (Deep Diabetic Retinopathy Image Dataset), v1.1
- Official archival record: <https://doi.org/10.5281/zenodo.8248825>
- Upstream repository: <https://github.com/deepdrdoc/DeepDRiD/tree/v1.1>
- Archive: `deepdrdoc/DeepDRiD-v1.1.zip`
- Published size: `1,373,472,897` bytes
- Published MD5: `3379e2fd7a2dd398545a67148420a5d3`
- License in the upstream v1.1 repository: Creative Commons
  Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

The verified extraction contains 2,279 files totaling 1,422,064,294 logical
bytes: 2,000 regular fundus JPEGs used for quality assessment, 256
ultra-widefield JPEGs not used by RetinaReady, and the upstream labels,
documentation, and license. The 2,000 regular images total 1,102,298,957
bytes (mean 551,149 bytes; range 290,979–2,530,590 bytes). Their native
dimensions are heterogeneous: 1,294 are 1736×1824; 660 are 1976×1984; and
the remaining 46 use four less-common dimensions. Resize/crop only inside the
training transform and retain the originals for reproducibility.

Zenodo describes the release as open access but uses the generic metadata
identifier `other-open`; the repository's included `LICENSE` file contains the
CC BY-SA 4.0 legal text. Preserve attribution, indicate modifications, and
apply the license's share-alike terms when redistributing adapted dataset
material. This repository does not redistribute the raw images.

## Citation

If using the dataset, cite:

> Ruhan Liu, Xiangning Wang, Qiang Wu, et al. "DeepDRiD: Diabetic
> Retinopathy—Grading and Image Quality Estimation Challenge." *Patterns*
> 3 (2022), 100512. <https://doi.org/10.1016/j.patter.2022.100512>

## Quality-label schema

Only the image-quality labels are included in RetinaReady manifests. Disease
grades are intentionally excluded because RetinaReady is a technical capture
quality tool, not a diagnostic system.

| Field | Upstream meaning |
| --- | --- |
| `overall_quality=0` | Not good enough for retinal-disease diagnosis |
| `overall_quality=1` | Good enough for retinal-disease diagnosis |
| `artifact` | `0` means none; `1/4/6/8/10` represent increasing affected area/severity |
| `clarity` | `1/4/6/8/10`; higher means progressively more vessels/lesions are identifiable |
| `field_definition` | `1/4/6/8/10`; higher means the optic disc and macula are better included and centered |

The derived `quality_label` maps `1` to `READY` and `0` to `RETAKE`. DeepDRiD
does not provide a third `LIMITED` class, so the preparation code does not
invent one.

## Official split policy

The manifests preserve the dataset's official split boundaries:

- `train`: `regular-fundus-training`
- `val`: `regular-fundus-validation`
- `test`: `Online-Challenge1&2-Evaluation`, using the subsequently published
  `Challenge2_labels.xlsx`

Each image ID begins with its patient ID and each patient has four images. The
preparation script verifies that no patient occurs in more than one manifest
and fails rather than producing a leaky split. The test split was opened once
for the earlier frozen Gemma smoke adapter. The specialist trainer refuses
`test.csv`, and no further model or threshold iteration should use it.

The verified v1.1 manifests contain:

| Split | Patients | Images | `RETAKE` | `READY` |
| --- | ---: | ---: | ---: | ---: |
| train | 300 | 1,200 | 624 | 576 |
| validation | 100 | 400 | 218 | 182 |
| test | 100 | 400 | 220 | 180 |

These labels describe technical sufficiency, not whether an eye is healthy.

## External MSHF stress data

`ml/evaluate_external_mshf.py` can evaluate the frozen specialist on the
author-provided MSHF test directory. The release is CC BY 4.0, contains 1,302
images across conventional CFP, portable-camera, and UWF sources, and is
downloaded locally from <https://doi.org/10.6084/m9.figshare.21507564>. The
expected `MSHF dataset 2.0.zip` MD5 is
`58203e6c2e064dafc800f5b83887487b`. Raw external data is ignored by version
control. The MSHF test was opened once on 2026-07-31; do not use it for tuning.
