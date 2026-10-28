# Application and engineering evidence

The [README](../README.md) focuses on the model and its results. This page collects the serving details and links to the original verification records. These are engineering checks, not medical-accuracy evaluations.

FastAPI and Gradio share a bounded queue and a persistent CPU worker, with archive validation, timeouts/restart, job access tokens and temporary-file cleanup. The viewer shows sampled input crops, 12 uncalibrated scores, warnings, timing and model provenance. Job tokens are not a complete public-service authentication system. [Architecture](ARCHITECTURE.md) and [local usage](LOCAL_USAGE.md) describe the contracts and repeat commands.

The [three-study Windows/Linux replay](MULTI_STUDY_VERIFICATION.md) covers complete and missing sequence slots across 464 provided DICOM files. Processed pixels and masks matched exactly; all scores were within **1.20e-7** of the private Kaggle reference, below the original 1e-5 tolerance. This checks the same checkpoint and input pipeline across deployment paths; it does not reproduce a competition leaderboard score from three cases.

Observed Linux worker processing was **3.871–10.140 seconds**, with a largest API-container cgroup peak of **1.13 GiB** under two CPUs / 2 GiB. These shared-machine observations exclude upload, queue wait and model startup. The memory scope includes the API process, worker, probe and attributed cache/tmpfs; it excludes Docker/WSL VM overhead, the host/browser and the separate direct verifier. These are not hosting-capacity estimates.

[Container checks](CONTAINER_VERIFICATION.md) cover non-root, read-only and network-disabled inference. The [final release review](RELEASE_NOTES.md) records 70 passing tests with authorized weights and 67 passing / 3 skipped in a clean public clone without them. That clone reused the existing dependency environment. Tests include generated-data inference; real MRI assets remain private.

Detailed evidence: [expanded replay report](multi_study_report.json), [verification history](BUILD_STATUS.md), [model card](MODEL_CARD.md), and [portfolio scope](PORTFOLIO.md). No application, checkpoint or resource measurement changed when this documentation was moved out of the README.
