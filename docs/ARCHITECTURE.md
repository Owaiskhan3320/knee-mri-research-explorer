# Architecture

One local Docker-compatible application, two Python processes: a FastAPI/Gradio process and a persistent CPU model worker. All UI and API submissions use one bounded JobManager queue. Three waiting jobs, one active study. Startup and per-job deadlines are enforced by the parent; a timed-out worker is terminated and replaced. The model loads once per worker, never per polling request.

Input: one study ZIP with `StudyUID/SeriesUID/slice.dcm`, plus a five-column metadata CSV. Exactly one study, supported single-frame DICOMs. At most 150 MB uploaded, 512 MB expanded, 4000 files, 16 MB per slice, 100 series. Reject traversal, symlinks, encrypted entries, duplicate paths and identifier mismatches before extraction.

Frozen pipeline: five plane/fat-suppression slots, eight 3-slice windows per slot, 140 mm centre crop, original float PIL bilinear resize, p1/p99 series scaling, uint8 rounding. Preserve failed-slice positions and missing-window masks. ConvNeXt-Tiny encoder, 256-dimensional projection and slot embedding, gated finding-specific attention, 12 logits and sigmoid. No report is required at inference. No batch percentile ranks, automatic diagnosis or calibration claim.

States: queued -> validating -> preprocessing -> inferencing -> completed; failed/cancelled terminal paths. Tokens grant job-specific access. Completed results expire after 30 minutes; explicit clear, error and shutdown remove generated study data. Jobs are not durable across restarts. Logs exclude image headers and access URLs. UI previews show sampled model-input crops, not a full-resolution medical viewer or lesion map.

Default host is 127.0.0.1. Local uploads are enabled; Docker defaults them off. No account, remote database, external API, telemetry, payment or hosting integration is configured. Dependencies are obtained from official free registries during setup only. Model assets are mounted read-only in Docker.

Synthetic example: generated mathematical intensity phantom in DICOM format. No competition images or patient data. Its predictions test pipeline mechanics and cannot establish model accuracy, medical utility or generalization.
