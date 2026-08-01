# Dataset-video guidance roadmap

## Implemented recording-loader boundary

The browser now has two intentionally different acquisition demonstrations:

1. **Start replay** remains the reliable, retrospective DeepDRiD sequence. It
   can hand the untouched pinned final still to the verified quality-first
   workflow.
2. **Open camera recording** decodes a prerecorded clinical-device MP4/WEBM/MOV
   locally and runs only lightweight frame telemetry. On five stable frames it
   freezes one local JPEG and, only under the final launcher's explicit
   experimental opt-in, sends that still through the quality-first workflow.
   Raw video bytes never enter either trained model.

The second path is tested with two openly licensed recordings. Xincheng Yao's
[moving color fundus video](https://opticapublishing.figshare.com/articles/media/Ultra-wide_field_video_fundus_photography/10728089)
provides a genuine positive retinal stream from a purpose-built PedCam. The
[Kestrel 3100m self-service recording](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935825/)
provides conventional tabletop B2B workflow context but exposes the operator
and device UI, not a raw color retinal stream. The modality guard therefore
rejects every sampled Kestrel frame as “No color fundus field detected.”

This distinction matters for a production B2B integration: many conventional
systems use an infrared alignment preview and emit a color still after flash
capture. A true live integration needs a manufacturer's supported preview SDK
or device stream. The browser recording loader proves the UI/controller and
failure boundary; it does not claim access to a camera's internal feed.

The PedCam recording is ultra-wide-field contact-camera footage, while the
trained still models target conventional central-field color fundus images.
It is consequently loader/telemetry QA only—not evidence that the still models
generalize to its frames.

## Decision

Build a **retrospective, dataset-video capture replay** for the hackathon. It
should show how a future fundus-device preview could guide recapture, but it
must not be presented as a live patient camera or a clinically validated
acquisition system.

This is achievable without retraining the current models. It is a small
controller/UI addition around the existing still-image contract:

```text
dataset-video / simulated preview
  -> inexpensive frame metrics + temporal controller
  -> select the best stable still frame
  -> existing RetinaReady still-image gate
  -> existing RetinaPriority handoff only when READY
```

Gemma is deliberately absent from the preview loop. It runs only after the
still-image gate accepts a frozen frame for optional review-priority routing.

## Scope for the next 3–4 hours

### Build

- A `Guided capture — retrospective replay` mode which renders a 20–30 second
  dataset-derived sequence at normal video speed.
- A single, stable instruction at a time: `Center retina`, `Hold still`,
  `Improve focus`, `Reduce glare`, `Improve illumination`, or `Capturing best
  frame`.
- A quality/stability indicator and a visible rolling “best frame” candidate.
- Autocapture only after the preview has been sufficiently stable; then send
  the selected still through the unchanged quality-first workflow.
- A visible provenance label: `Retrospective dataset replay — no live camera
  or clinical validation`.

### Do not build

- Webcam, bare-phone-camera, pupil-dilation, or patient-facing camera capture.
- Millimeter commands such as “move left 2 mm.” Those require calibrated
  device geometry and hardware-specific validation.
- Continuous Gemma/VLM calls, diagnosis, pathology localization, or a claim
  that the preview controller is a clinical decision system.
- A new model or LoRA training run.

### Best demo asset strategy

The current verified specialist is intentionally restricted to four exact
DeepDRiD image hashes. Synthetic transforms of those images do **not** pass the
existing gate. Therefore the fastest truthful implementation is:

1. Use local DeepDRiD stills to render a simulated acquisition sequence
   (controlled crop/translation, blur, illumination, or glare transitions).
2. Drive preview guidance from the known transform/cheap frame metrics.
3. At autocapture, pass the unmodified, pinned DeepDRiD `READY` frame into the
   existing still-image workflow. The verified `READY -> RetinaPriority`
   handoff remains intact.

This separates a clearly labelled demonstration controller from the verified
still-image gate rather than silently expanding the currently fixed-input
model scope.

The public [RVD dataset](https://zenodo.org/records/8287928) is a strong later
asset for a genuine video replay: it contains 635 smartphone-based videos at
25 FPS, lasting 2–30 seconds. It is not a quick download for the hackathon:
the archive is 26.1 GB and split into 756 MB–3.8 GB ZIP files. More
importantly, its annotations concern vessels and pulsation, not recapture
reasons or corrective actions. Do not train on it today and do not claim it
supports learned commands.

## Preview controller

### Frame metrics

At 4–6 Hz, compute simple local measurements from the newest rendered frame.
The video itself can continue at 25–30 FPS.

| Signal | Simple demo measurement | Guidance when poor |
|---|---|---|
| Retinal field / centering | bright-retinal-region bounding box, crop coverage, centre offset | `Center retina` / `Adjust framing` |
| Sharpness | variance of Laplacian or Tenengrad | `Improve focus` |
| Motion | frame-to-frame difference or optical-flow magnitude | `Hold still` |
| Exposure | luminance mean and clipped-black percentage | `Improve illumination` |
| Glare / saturation | clipped-bright-region percentage, with a conservative threshold | `Reduce glare` |
| Stability | a recent-window aggregate of the above | `Hold position` / `Capturing best frame` |

These are **capture-quality heuristics**, not findings about the retina. The
preview UI must avoid overlaying a pathology heatmap or describing any medical
condition.

### Instruction precedence and smoothing

Use this priority order so the screen never flickers between competing advice:

1. No usable retinal field / severe framing problem.
2. Motion or severe blur.
3. Extreme glare or illumination problem.
4. Stable enough to capture.

Smooth each score with a short moving average or EMA. Keep an instruction until
the replacement condition persists for at least two analysis ticks. This is
more legible than exposing fluctuating raw probabilities.

The design follows existing evidence that capture quality can be decomposed
into position, illumination, and clarity. The published DeepFundus system
provides multidimensional quality classification and real-time acquisition
guidance, while the Vistaro camera uses a frame decision tree and requires a
run of correct frames before autocapture. [DeepFundus](https://pmc.ncbi.nlm.nih.gov/articles/PMC9975093/)
[Vistaro autocapture study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8525841/)

### Autocapture state machine

```text
PREVIEW
  -> GUIDE (one dominant technical instruction)
  -> STABILIZING (quality good across a short rolling window)
  -> CAPTURE (freeze the best-scoring frame from that window)
  -> STILL_QUALITY_GATE
       READY   -> optional RetinaPriority -> clinician review queue
       RETAKE  -> back to GUIDE with retake instruction
       LIMITED -> human review / no priority release
```

Suggested hackathon thresholds:

- Analyze no more than once every 200–250 ms.
- Require 4–5 consecutive passing analysis ticks before autocapture.
- Preserve the highest score from the preceding one-second window, rather than
  blindly taking the last frame.
- Fire capture once, disable additional captures while the still gate runs,
  and fail closed if anything malformed or unknown occurs.

This is consistent with deployed/researched systems that perform quality
control before clinical workflow entry and prompt a recapture below a quality
threshold. [On-device retinal camera study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12289297/)

## Workflow handoff

The current product boundary remains unchanged:

```text
Retrospective replay controller
  -> frozen candidate still
  -> RetinaReady: READY / RETAKE / LIMITED
  -> only READY reaches RetinaPriority
  -> PRIORITY_REVIEW / ROUTINE_REVIEW / UNCERTAIN
```

In particular:

- `RETAKE`, `LIMITED`, decode failure, and controller failure never invoke
  Gemma or release a review-priority suggestion.
- `ROUTINE_REVIEW` never means “normal” or “healthy.”
- `PRIORITY_REVIEW` is a non-diagnostic queue suggestion for clinician review.
- A preview-quality indicator is not a substitute for the existing still gate.
- Prerecorded device video enters this handoff only as one browser-frozen JPEG,
  tagged for the explicitly enabled experimental candidate route. Raw video is
  never submitted.

The quality specialist's measured local latency of roughly 123–153 ms is
compatible with a 4–6 Hz still-quality loop. The candidate opt-in deliberately
expands beyond its fixed dataset hashes for this one OOD demo path, so neither
its decision nor any downstream priority result is validation on video frames.

## Minimum acceptance criteria

Before showing it to judges, verify all of the following:

1. The replay visibly transitions from poor framing or instability to a stable
   capture-ready state.
2. Only one non-medical recapture instruction is displayed at any moment and
   it changes smoothly.
3. The UI selects and freezes a best frame exactly once after the stability
   window, rather than automatically submitting every frame.
4. The frozen frame visibly enters the existing still-image quality gate.
5. `RETAKE` and `LIMITED` demonstrably block RetinaPriority; only `READY`
   reaches it.
6. Gemma makes zero calls while the preview is running.
7. The UI visibly says `retrospective dataset replay` and contains no claim of
   camera validation, live patient capture, diagnosis, physical distance, or
   clinical clearance.
8. A backup 20–30 second screen recording is available in case of local demo
   failure.

## Evidence and claim boundary

The underlying product direction is credible: a recent smartphone-video study
trained an EfficientNet-B0 to select diagnosable frames from fundus videos,
showing that frame-level selection is a real upstream quality-control problem.
It is a 2026 preprint with a small device-specific dataset, so it supports
feasibility—not a performance claim for RetinaReady. [Study record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6721376)

There is also precedent for immediate quality feedback: the Vistaro study
reported autocapture in 80% of examinations in about 10–15 seconds and 91.6%
of resulting images as clinically useful. That system used specialized
optics/illumination and its own device calibration; it does **not** validate a
bare phone or this project. [Vistaro study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8525841/)

The accurate claim is:

> RetinaReady demonstrates a local, dataset-replay capture controller that can
> choose a stable candidate frame, apply a verified still-image quality gate,
> and only then offer non-diagnostic clinician-review prioritization.

## Post-hackathon path

1. Download and curate a small, documented RVD replay subset; use it for
   temporal UX and best-frame-selection testing, not supervised corrective
   labels.
2. Integrate a real fundus device's preview SDK or a validated attachment. A
   bare phone camera does not provide a reliable fundus stream.
3. Collect device-specific raw preview videos with final accepted frames,
   quality reason labels, operator interventions, and device/pupil metadata.
4. Validate the frame quality gate across operators, cameras, sites, and
   patient subgroups before making real acquisition claims.
5. Only then consider a learned guidance head. Millimeter directives require
   device calibration and a defined pixel-to-physical-distance mapping, not a
   Gemma prompt or generic LoRA.
