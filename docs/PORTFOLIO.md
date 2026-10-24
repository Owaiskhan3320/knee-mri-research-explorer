# Portfolio release and short demo

Source repository: https://github.com/Owaiskhan3320/knee-mri-research-explorer. This source release contains code, documentation and generated-data media; weights and real studies remain private. No paid API or public inference hosting is configured.

## What this project demonstrates

- Project-trained ConvNeXt-Tiny with finding-specific multiple-instance attention, using public report-derived weak labels and ImageNet initialization.
- DICOM ordering, physical crop/resize, explicit missing-window masks and frozen preprocessing provenance.
- A CPU inference worker shared by FastAPI and a Gradio viewer, bounded queue, timeout/restart, access tokens and temporary-file cleanup.
- An honest model card, 18 passing engineering tests and exact real-study Windows/Kaggle input parity with near-identical scores.

The model deployed here scored 0.833 public macro ROC AUC. The separate 0.950 public-model reproduction is not this model and should not be presented as independently trained work. One-study parity is not evidence of clinical accuracy.

## Before a GitHub release

1. Inspect `git status` and the exact staged file list. Keep `.venv/`, `.runtime/`, `data/`, weights and credentials excluded.
2. Use only generated-phantom screenshots/video. A real MRI screenshot is also dataset content and is kept private.
3. Review the original report-label source attribution and relevant competition/MIRA and pretrained-weight terms before distributing any data or weights. The app does not download weights automatically.
4. Keep Docker explicitly marked untested until it has been built and run locally. The existing Python installation is the verified launch path.
5. Publish code/documentation only after that review. A public cloud inference service is a separate release decision; no paid service is required for a local demo.

## Suggested 60–90 second recording

A [60-second captioned walkthrough](demo.mp4) and [WebVTT captions](demo.vtt) are included. It uses nine actual browser captures of generated-data interactions, held for readability, with no narration. It is not a continuous or real-time performance recording. The outline below can also guide a longer narrated recording.

| Time | Show | Explain |
| --- | --- | --- |
| 0–10 s | Title, local CPU badge and research notice | A local research demo for 12 study-level knee MRI findings |
| 10–30 s | Generated software test → Run local analysis | Artificial DICOMs test the pipeline; these scores have no medical meaning |
| 30–45 s | Change input slot and sampled window | The viewer shows actual cropped model inputs, not lesion heatmaps |
| 45–60 s | Twelve scores and JSON download | Outputs include model identity, valid windows, warnings and timing |
| 60–75 s | Model card and verification table | Own model: 0.833 public AUC; real-study pixels match Kaggle, with score difference below 6e-8 |
| 75–90 s | Clear study | Bounded local processing and temporary-file cleanup |

Do not show private file paths containing study identifiers, real scans, result access tokens or account credentials in a public recording.

## Defensible resume wording

“Built a local knee MRI research application with a project-trained ConvNeXt-Tiny attention-MIL classifier, DICOM preprocessing, FastAPI and an interactive viewer; achieved 0.833 public macro ROC AUC and verified exact Windows/Kaggle preprocessing parity on a real study.”

Use the engineering details you can explain in an interview. Add latency only with the one-study/CPU measurement limitation; do not describe the application as clinically validated or production-ready.
