# Recent retinal image-quality research translated into RetinaReady

This note separates what the cited papers actually establish from the
engineering ideas RetinaReady borrows. Headline metrics across datasets,
splits, label definitions, and operating points are not directly comparable.

## Highest-leverage findings

| Work | Reported idea/result | RetinaReady translation | Saturday status |
| --- | --- | --- | --- |
| [Telesco et al., 2026](https://doi.org/10.1016/j.bspc.2025.109167) | Jointly learning overall quality with illumination, clarity, and contrast improved the authors' DeepDRiD F1 from 0.763 to 0.778 and recall from 0.782 to 0.845. [Labels/code](https://github.com/ltelesco/Semi-Supervised-Multi-Task-Learning-for-Interpretable-Quality-Assessment-of-Fundus-Images) are public. | Keep overall quality and acquisition-factor heads in one specialist. The next accuracy challenger should unfreeze the final DenseNet block; another frozen head-only rerun is unlikely to help. | Multi-task frozen-feature model deployed; encoder-unfreezing challenger pending A100/local compute. |
| [SAM-IQA, 2026](https://doi.org/10.1016/j.exer.2026.110938) | Combines a global image branch with salient local evidence and reports 0.815 DeepDRiD AUC in its protocol. | Test global pooled features plus a small spatial grid/local representation. Treat this as an approximation, not a reproduction. Promote only if it improves a fixed-risk metric on internal patient splits. | The 2x2 spatial challenger finished and did not pass promotion; the deployed global artifact remains frozen. |
| [EFIQA, MIDL 2026](https://proceedings.mlr.press/v315/wang26f.html) | Learns spatial image-quality evidence from vessel-masked inpainting and distills it into a small adapter over frozen features. [Code](https://github.com/penway/EFIQA) is public. | Add an explanation-only quality-attention overlay. It may help an operator see where the model found degraded capture quality, but it must never promote READY and must be labeled as non-pathology localization. | Lightweight factor-specific Grad-CAM overlay implemented; full EFIQA reproduction deferred. |
| [FTHNet/FQS, 2025](https://www.nature.com/articles/s41598-025-24423-8) | Uses continuous expert quality scores rather than only discrete labels; releases FQS with 2,246 images and a compact model. [Code](https://github.com/HudenJear/BasiQA) and [data](https://figshare.com/articles/dataset/FIQS_Dataset_Fundus_Image_Quality_Scores_/28129847) are public. | A future risk/workload slider could expose continuous quality while decisions remain thresholded and selective. FQS criteria must be mapped before mixing labels. | Product direction only; no data mixing before label audit. |
| [Swin-MCSFNet, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996994/) | Fuses RGB, HSV, and LAB representations and includes spatial score maps; the authors report higher EyeQ accuracy than a single-view Swin baseline. | Test color/spatial fusion as a reversible challenger. Extra views increase runtime and can overfit, so deployment remains on the single RGB artifact until a patient-grouped comparison wins. | Spatial-plus-color challenger finished and was rejected: lower AUC, balanced accuracy, and coverage. |
| [LGAANet, 2024](https://doi.org/10.3389/fmed.2024.1418048) | Preserves local and global spatial evidence rather than collapsing immediately to one pooled vector. | Retain a small 2x2 feature grid in the challenger and use the map only for quality explanation. | Tested as a frozen-feature approximation; not promoted. |
| [Multi-center refined CFP quality model, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10903193/) | Models overall quality with structured factors such as clarity, artifact, optic-disc/macula position, laterality, and field of view. | Keep recapture guidance tied to explicit acquisition factors rather than unconstrained VLM prose. | Implemented for artifact, clarity, and field definition; location/laterality not claimed. |

## What the current evidence says

The deployed specialist's exploratory DeepDRiD validation AUC is 0.803 at the
ordinary ranking threshold. Its frozen three-way policy covers 11.0% of
specialist-only validation images at 93.18% accepted accuracy. In the complete
local Gemma pipeline, coverage is 10.75% and accepted accuracy is 93.02%; Gemma
vetoes or abstains once and never promotes a specialist abstention.

The external MSHF stress test is more informative than another internal tenth
of a point. The small standard central-field CFP group retains useful ranking
(AUC 0.784) and all 12 accepted decisions are correct; this is encouraging, not
validation. Portable images receive zero automated coverage, so the test says
nothing about safety or usefulness there. Ultra-widefield mosaics fail to
transfer and account for all four external false READY calls. Therefore:

1. keep the product scope at conventional central-field color fundus photos;
2. treat portable input as an unproven future cohort;
3. keep the completed DeepDRiD UWF veto experiment outside the runtime until a
   separately sourced multi-device cohort validates it; never tune on MSHF;
4. optimize **coverage at a fixed false-READY constraint**, not raw accuracy;
5. preserve the complete 400-image hybrid run as the current end-to-end
   reference and keep all abstentions/failures in every future denominator.

The first veto-only UWF experiment is now complete using only DeepDRiD modality
data. It detected 50/50 official-validation UWF images and falsely vetoed 4/400
conventional images, with AUC 1.000. A paired same-patient holdout achieved
clean separation for 23/25 patients. Those results are compelling in-domain,
but the small cohort and perfect ranking suggest a possible camera/domain
shortcut. The artifact therefore remains explicitly unauthorized for runtime
integration. See `docs/UWF_VETO_EXPERIMENT.md`.

## Completed spatial/color challenger

The reversible frozen-feature experiment compared the deployed global pooled
representation with (a) a global plus 2x2 spatial grid and (b) that spatial
representation plus fixed color statistics. It reused the same patient-grouped
fit, tuning, and calibration protocol and did not touch DeepDRiD test or MSHF.

| Variant | Validation AUC | Balanced accuracy | Selective coverage | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: |
| Global baseline | **0.803** | **74.20%** | **11.00%** | 93.18% |
| Global + 2x2 spatial | 0.792 | 73.19% | 8.75% | **97.14%** |
| Global + 2x2 spatial + color | 0.791 | 72.14% | 8.25% | 90.91% |

The spatial variants slightly reduced mean acquisition-factor error, but both
lost ranking quality and automated coverage. The higher accepted accuracy of
the spatial-only row came from making fewer decisions; it is not a broad model
improvement. The promotion rule therefore retained the global baseline. Full
machine-readable results are in `outputs/spatial-color-challenger/report.json`.

## Promotion rule for any challenger

A paper-inspired challenger is not better merely because its validation AUC is
higher. Freeze it before evaluation and require all of the following:

- patient-grouped fitting and threshold setting;
- no use of DeepDRiD test or the opened MSHF test for selection;
- higher ROC-AUC or materially higher selective coverage at the same
  false-READY target;
- no worse acquisition-factor error;
- valid READY/RETAKE/LIMITED behavior through the real local API;
- no new network dependency or unacceptable 24 GB Mac latency;
- fresh-patient calibration before any future risk guarantee.

## Practical sequence

1. Use the implemented explanation-only attention overlay in the live story;
   it does not touch decisions. Call the factor “predicted,” and do not claim
   causal localization; the 23-case presentation audit is documented.
2. Keep the completed global/local/color result as a documented negative
   experiment; it failed the promotion rule and must not alter the demo model.
3. When A100 access returns, unfreeze DenseNet's final block for a small
   multi-task specialist challenger; train several seeds and recalibrate only
   after model selection.
4. Run the already prepared all-1,200-image Gemma LoRA configuration as a
   separate language/multimodal challenger.
5. Freeze one final pipeline and collect a new, independently sampled
   prospective cohort. That step—not another post-hoc threshold—is what turns
   the risk-control math into a defensible claim.
