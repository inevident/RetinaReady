# RETAKE explanation presentation audit

This is a presentation-safety audit, not a clinical explanation study.

The refreshed 400-image hybrid run produced 23 final RETAKE decisions. All 23
generated a valid local factor Grad-CAM PNG; 22 were true RETAKE labels and one
was the known false RETAKE. READY and LIMITED produced no map.

For all 23 cases, the specialist's lowest predicted factor was artifact quality,
so every overlay targeted the artifact head. Artifact was also the lowest or
tied-lowest released DeepDRiD factor label in only 6/23 cases (26.1%). This
comparison is exploratory and the factor label scales are coarse, but it is
enough to reject a stronger claim that the overlay reliably identifies the true
cause of poor capture quality.

Decision for the hackathon:

- retain the overlay as **model quality attention—not pathology localization**;
- label the text as the **weakest predicted factor**;
- show only the curated obvious RETAKE example during the live demo;
- do not claim lesion localization, defect segmentation, or causal explanation;
- keep the overlay outside the gate so it cannot alter READY/RETAKE/LIMITED;
- treat factor-specific instruction accuracy as unvalidated future work.

The machine contract for presence/absence is recorded in
`outputs/hybrid-validation-exploratory.json`. The factor head's aggregate MAE
is documented separately in `docs/QUALITY_SPECIALIST_MODEL_CARD.md`.
