# Linux real-study parity and initial CPU benchmark — 25 October 2026

**Passed on the same single authorized study previously verified on Windows.** This adds a Linux deployment check and repeated-request observations; it adds no independent studies and does not measure medical accuracy.

An existing local Linux/amd64 image ran on Docker Desktop/WSL2 with a two-CPU quota, 2 GiB memory limit and container swap disabled. Model and study folders were mounted read-only. Both temporary containers had network mode `none`; no GPU, remote service, download, model training or payment was used. Other applications on the shared host were not stopped.

## Consistency checks

| Check | Result |
| --- | --- |
| Source inputs | All 159 DICOM file hashes matched the private Kaggle reference |
| Processed images | Every uint8 input pixel matched the reference arrays and original cache identity |
| Window mask | Exact match: 32 valid of 40 windows; missing sequence remained masked |
| Targets | All 12 checked by name |
| Fixed tolerance | Maximum absolute score difference of 0.00001 |
| Direct Linux/Kaggle difference | Maximum 0.00000011920928955078125 |
| API Linux/Kaggle difference | Same maximum, on every request |
| Request consistency | All four API score vectors identical |
| Access and cleanup | All 32 previews available with token, unauthorized requests rejected, each job deleted |

The frozen model and preprocessing source were unchanged. The deployed model remains the project-trained ConvNeXt-Tiny epoch 8 with a **0.833 public competition macro ROC AUC**. No leaderboard score was recomputed.

## Observed timings

The API worker was already ready before the first request. “First request” does not mean a cold model start.

| Request | Worker preprocessing | Worker inference | Worker total | Local upload to result |
| --- | ---: | ---: | ---: | ---: |
| First | 0.779 s | 4.781 s | 6.951 s | 8.681 s |
| Subsequent 1 | 0.616 s | 4.062 s | 6.090 s | 7.490 s |
| Subsequent 2 | 0.642 s | 4.237 s | 6.253 s | 8.066 s |
| Subsequent 3 | 0.617 s | 4.143 s | 6.127 s | 8.106 s |

Subsequent worker-total median: **6.127 seconds**. Worker total includes validation and preview generation, and excludes upload, queue wait and model startup. Upload-to-result includes the local multipart transfer and polling. Readiness wait was 11.015 seconds from probe start; it is not a full Docker startup benchmark. Four requests on one case cannot establish tail latency or throughput under concurrent load.

## Memory measurement

API-container lifetime `memory.peak`: **1,022,607,360 bytes (975.234 MiB)**. The reading covers cgroup-accounted server/worker/probe memory and attributed filesystem cache/tmpfs, including startup and the four requests. It excludes the separate direct-parity container, Docker Desktop/WSL VM overhead, browser and Windows host. It is a different scope from the older Windows verifier-process working-set peak.

No `oom`, `oom_kill` or `max` memory events occurred; reported swap peak was zero. The 2 GiB limit is a tested configuration for this workload, not a minimum hosting specification. The [Linux kernel documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html) defines cgroup accounting and `memory.peak`; [Docker's resource documentation](https://docs.docker.com/engine/containers/resource_constraints/) explains CPU, memory and swap limits.

## Repeat privately

With the local image built and authorized model/replay assets already available:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\benchmark_container.py' 'data\local_replay'
```

The script runs direct parity in one temporary container and actual API requests in a fresh second container, so the direct verifier's model allocation is excluded from the API peak. Uploads are enabled only in that network-isolated test container; the app image's default upload policy remains off. Both containers are removed in cleanup. The script writes a sanitized report under `.runtime/linux_real_study_report.json`; the reviewed copy is [linux_real_study_report.json](linux_real_study_report.json). No scan, reference arrays, per-study identifier or weights are published.

The next validation step is additional authorized studies with different input-slot completeness, protocols and codecs. This result does not establish clinical utility, calibration or scanner/site generalization.
