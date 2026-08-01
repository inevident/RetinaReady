# Third-party notices

The repository-level MIT License applies to RetinaReady's original source code
and documentation. It does not replace the terms attached to third-party data,
media, pretrained parameters, or model-derived artifacts.

## DeepDRiD still images

The sample stills under
`demo-assets/color-fundus-sample-pack/stills/` come from DeepDRiD v1.1 and are
redistributed under CC BY-SA 4.0. The images are renamed and packaged for the
demo; their pixels are otherwise unmodified. Preserve attribution, indicate
later modifications, and share adaptations under compatible terms.

- Release: <https://doi.org/10.5281/zenodo.8248825>
- Repository: <https://github.com/deepdrdoc/DeepDRiD/tree/v1.1>
- Paper: <https://doi.org/10.1016/j.patter.2022.100512>
- License: <https://creativecommons.org/licenses/by-sa/4.0/>

## Fundus videos

The two clips under `demo-assets/color-fundus-sample-pack/videos/` are
redistributed under CC BY 4.0. Exact titles, creators, DOI links, intended demo
roles, and modification notes are recorded in the sample pack's `README.md`.

- Xincheng Yao, “Ultra-wide field video fundus photography”:
  <https://doi.org/10.6084/m9.figshare.10728089.v1>
- Zhang et al., “Evaluating imaging repeatability of fully self-service fundus
  photography …,” Additional file 1:
  <https://doi.org/10.1186/s12938-024-01222-2>
- License: <https://creativecommons.org/licenses/by/4.0/>

## TorchVision DenseNet-121 parameters

`models/retinaready-quality-specialist/densenet121-a639ec97.pth` contains the
TorchVision DenseNet-121 IMAGENET1K_V1 pretrained parameters used by the frozen
quality specialist. TorchVision is distributed under the BSD 3-Clause License:
<https://github.com/pytorch/vision/blob/main/LICENSE>. A copy is included at
[`licenses/TORCHVISION-BSD-3-CLAUSE.txt`](licenses/TORCHVISION-BSD-3-CLAUSE.txt).

## Gemma 4 and RetinaPriority LoRA

The Gemma 4 base weights and multimodal projector are not redistributed in
this repository. Their identifiers, revisions, hashes, conversion provenance,
and expected local paths are documented in `docs/LOCAL_TUNED_BUNDLE.md`.

The compact RetinaPriority LoRA under `models/retinapriority-gemma4-26b/` was
trained against Gemma 4 and is a modified, derived artifact. Its adjacent
`NOTICE` records the changes and pinned base revision. It must be used
consistently with Google's Gemma license and prohibited-use terms. See
<https://ai.google.dev/gemma/docs/gemma_4_license>; a copy of Apache License
2.0 is included at
[`licenses/GEMMA-4-APACHE-2.0.txt`](licenses/GEMMA-4-APACHE-2.0.txt).

## Research-only scope

These artifacts are supplied for a public hackathon research prototype. They
are not clinically validated and are not licensed or represented as a medical
device, diagnosis system, or treatment system.
