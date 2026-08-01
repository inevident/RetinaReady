# RetinaReady color-fundus sample pack

Open `index.html` for an offline visual gallery, or browse `stills/` and
`videos/` directly.

## What's included

- **4 gradable color-fundus stills** — filenames begin with `GRADABLE`.
- **4 ungradable color-fundus stills** — filenames begin with `UNGRADABLE`.
- **1 genuine moving color-fundus video** from a purpose-built PedCam.
- **1 conventional tabletop-camera workflow video** from a Kestrel 3100m.
- `labels.csv` with the exact upstream technical-quality annotations.
- `SHA256SUMS.txt` for integrity checks.

`GRADABLE` and `UNGRADABLE` are the upstream DeepDRiD technical-quality
labels. They do **not** mean healthy/diseased, routine/urgent, or
diagnosis/no-diagnosis.

## Which files work in the current app?

The current live research demo intentionally accepts only four exact image
hashes:

| File marker | Expected app behavior |
| --- | --- |
| `APP_ROUTINE` | Quality passes, then the fixed demo can show routine review routing. |
| `APP_PRIORITY` | Quality passes, then the fixed demo can show priority review routing. |
| `APP_LIMITED` | The conservative quality policy abstains; escalation is blocked. |
| `APP_RETAKE` | The quality policy requests recapture; escalation is blocked. |

Files marked `REFERENCE_ONLY` are extra, correctly labeled visual examples.
They are intentionally outside the app's four-hash inference scope. Do not
interpret a fail-closed app response on those files as a new model result.

The moving PedCam video can be selected with **Open camera recording**. It is
used only by the browser telemetry preview and never reaches the trained still
models. The Kestrel video shows the physical B2B workflow but not a raw color
retinal stream, so the preview should say **No color fundus field detected**.

## Still-image source and license

The stills are unmodified copies from DeepDRiD v1.1:

- Archive: https://doi.org/10.5281/zenodo.8248825
- Upstream repository: https://github.com/deepdrdoc/DeepDRiD/tree/v1.1
- License: CC BY-SA 4.0
- Citation: Ruhan Liu, Xiangning Wang, Qiang Wu, et al. “DeepDRiD: Diabetic
  Retinopathy—Grading and Image Quality Estimation Challenge.” *Patterns* 3
  (2022), 100512. https://doi.org/10.1016/j.patter.2022.100512

Preserve attribution and the CC BY-SA 4.0 terms if you redistribute these
stills. No images in this pack have been modified.

## Video sources and licenses

### Moving color retinal video

- “Ultra-wide field video fundus photography,” Xincheng Yao, 2019
- Source: https://opticapublishing.figshare.com/articles/media/Ultra-wide_field_video_fundus_photography/10728089
- DOI: https://doi.org/10.6084/m9.figshare.10728089.v1
- License: CC BY 4.0

This is genuine color-fundus video from a purpose-built contact-mode pediatric
fundus camera. It is not a smartphone recording and is not evidence that the
central-field still models generalize to this device.

### Conventional tabletop workflow video

- Zhang J. et al., “Evaluating imaging repeatability of fully self-service
  fundus photography within a community-based eye disease screening setting,”
  Additional file 1, 2024
- Source: https://doi.org/10.1186/s12938-024-01222-2
- License: CC BY 4.0

This recording shows a Kestrel 3100m non-mydriatic tabletop camera in use. It
is acquisition B-roll, not a raw retinal video feed. No author, dataset, or
device endorsement is implied.

## Safety boundary

These are research/demo assets, not clinical validation. Never describe a
gradable image as healthy, an ungradable image as diseased, or a browser
telemetry score as a medical decision.
