# Portfolio release and short demo

Source repository: https://github.com/Owaiskhan3320/knee-mri-research-explorer. This source release contains code, documentation and generated-data media; weights and real studies remain private. No paid API or public inference hosting is configured.

## What this project demonstrates

- Project-trained ConvNeXt-Tiny with finding-specific multiple-instance attention, using public report-derived weak labels and ImageNet initialization.
- DICOM ordering, physical crop/resize, explicit missing-window masks and frozen preprocessing provenance.
- A CPU inference worker shared by FastAPI and a Gradio viewer, bounded queue, timeout/restart, access tokens and temporary-file cleanup.
- A model card, [70 passing application/research tests](RELEASE_NOTES.md) and exact Windows/Linux preprocessing parity on three authorized studies, covering 464 provided DICOM files and complete/missing sequence sets.
- A locally verified Linux CPU container: non-root, read-only model/root filesystem, missing-model rejection and inference with networking disabled.

The model deployed here scored 0.833 public macro ROC AUC. The separate 0.950 public-model reproduction is not this model and should not be presented as independently trained work. These three parity cases verify implementation consistency; they do not measure clinical accuracy.

## Before a GitHub release

1. Inspect `git status` and the exact staged file list. Keep `.venv/`, `.runtime/`, `data/`, weights and credentials excluded.
2. Use only generated-phantom screenshots/video. A real MRI screenshot is also dataset content and is kept private.
3. Review the original report-label source attribution and relevant competition/MIRA and pretrained-weight terms before distributing any data or weights. The app does not download weights automatically.
4. Repeat `scripts/verify_container.py` after changing the container or application. The [expanded record](MULTI_STUDY_VERIFICATION.md) covers three private studies on Windows/Linux, eight Linux API requests and cgroup peak memory. Wider scanner/protocol and compressed-study coverage remain open.
5. Publish code/documentation only after that review. A public cloud inference service is a separate release decision; no paid service is required for a local demo.

## Suggested 60–90 second recording

A [60-second captioned walkthrough](demo.mp4) and [WebVTT captions](demo.vtt) are included. It uses nine actual browser captures of generated-data interactions, held for readability, with no narration. It is not a continuous or real-time performance recording. The outline below can also guide a longer narrated recording.

| Time | Show | Explain |
| --- | --- | --- |
| 0–10 s | Title, local CPU badge and research notice | A local research demo for 12 study-level knee MRI findings |
| 10–30 s | Generated software test → Run local analysis | Artificial DICOMs test the pipeline; these scores have no medical meaning |
| 30–45 s | Change input slot and sampled window | The viewer shows actual cropped model inputs, not lesion heatmaps |
| 45–60 s | Twelve scores and JSON download | Outputs include model identity, valid windows, warnings and timing |
| 60–75 s | Model card and verification table | Project-trained model: 0.833 public AUC; three real-study replays matched inputs exactly, with scores within 1.20e-7 of Kaggle |
| 75–90 s | Clear study | Bounded local processing and temporary-file cleanup |

Do not show private file paths containing study identifiers, real scans, result access tokens or account credentials in a public recording.

## Resume wording

Project title: **Knee MRI Research Explorer — PyTorch, ConvNeXt, FastAPI, Gradio, Docker**. Link the [repository](https://github.com/Owaiskhan3320/knee-mri-research-explorer) and [generated-data demo](demo.mp4).

- Trained a ConvNeXt-Tiny attention-MIL classifier for 12 study-level knee MRI findings using public report-derived weak labels and ImageNet initialization; achieved **0.833 public macro ROC AUC** in the RSNA Knee Abnormality Detection competition.
- Built a local CPU application with validated DICOM uploads, an isolated inference worker, bounded queue, job access tokens and cleanup; verified exact Windows/Linux preprocessing parity across **three authorized studies (464 DICOM files)** and tested offline Docker inference with read-only model mounts.

Compact alternative: “Built a local knee MRI research app with ConvNeXt attention-MIL, FastAPI, Gradio and Docker; achieved 0.833 public macro ROC AUC and verified matching Windows/Linux preprocessing on three real studies.”

Use the measured details you can explain. Timing and memory belong with their [sample and scope](MULTI_STUDY_VERIFICATION.md), and are optional on a resume. Lead with the trained model and working application. A rank from the separate public-model reproduction should be attributed separately rather than attached to this classifier.

## Interview walkthrough

Explain the project in this order: the 12 study-level findings; report-derived weak supervision; five sequence slots and three-slice windows; finding-specific attention and missing-window masks; the CPU worker/API/viewer; then evidence and limitations.

Be ready to explain these choices:

- Pilkwang Kim published the report-derived labels. This project used those outputs, retained YES/NO soft scores and masked UNK labels; it did not build a new multilingual report extractor.
- ImageNet initialized the encoder. The project trained the MRI classifier and MIL components; it did not train ConvNeXt from random initialization.
- Macro ROC AUC measures ranking across findings. The value 0.833 is not 83.3% diagnostic accuracy or a calibrated patient risk.
- Hashes, exact arrays and fixed score tolerance test deployment consistency. Three selected studies and a reused 58-study expert development set do not establish clinical generalization.
- Tokens protect individual local jobs. Queue limits, timeout/restart, temporary-file expiry and read-only container mounts serve specific failure and data-handling needs.
- The public artifact is source, documentation and a generated-data recording. Private weights and scans stay local; no paid or public inference host is needed to demonstrate this work.

The included video records the earlier generated-data UI walkthrough. The expanded verification record is the current evidence for three-study parity; no real MRI footage is published.
