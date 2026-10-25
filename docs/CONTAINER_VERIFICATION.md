# Local Docker verification — 25 October 2026

Status: **passed**. This record covers local deployment engineering with generated DICOM inputs. It is not a new model-accuracy evaluation.

The image was built using Docker Desktop 4.72.0, Engine 29.4.2 and the local `desktop-linux` builder/context on Windows with WSL2. The image platform is `linux/amd64`; the base is `python:3.12-slim-bookworm`. No image was pushed, no cloud builder or GPU was used, and no payment was made.

## Checks performed

| Check | Result |
| --- | --- |
| Dependencies | `pip check` passed; torch 2.11.0+cpu and torchvision 0.26.0+cpu |
| DICOM codecs | JPEG Lossless and JPEG 2000 decoders available |
| Image contents | All application source hashes match the checkout; no baked-in weights or runtime files |
| Missing weights | Liveness 200, readiness 503, inference submission rejected with 429 |
| Offline inference | Container network mode `none`; actual model returned 12 scores and 40 valid windows for the generated phantom |
| Access and cleanup | Unauthenticated status/preview requests rejected; authorized result/preview access and job deletion passed |
| Default upload policy | Upload request rejected with 403 |
| Runtime restrictions | UID 1000, read-only root filesystem and weights, temporary writable tmpfs directories |
| Resource limits | Two CPUs, 2 GiB memory; no OOM termination during these checks |
| Host access | Readiness reached from Windows; published port bound only to 127.0.0.1 |
| Test cleanup | Temporary test containers removed; local image and model files retained |

Exact image identity, timings and post-request memory snapshots are in [container_verification.json](container_verification.json). These snapshots are not peak usage or a capacity estimate. The image contains no checkpoint; authorized private weights are mounted at runtime. Build-time dependency downloads are required for a cold build; the offline inference phase needs no network.

## Repeat locally

Use the commands in the [README](../README.md#docker-optional-local-cpu) to build the image. With the authorized weights already imported into `models/`, run:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\verify_container.py' --context desktop-linux --image knee-mri-explorer:local --port 7861
```

The script refuses remote TCP Docker contexts, checks that the existing image matches the application source, runs three temporary containers, and writes `.runtime/container_verification.json`. It does not build, push, download data or use a GPU. Port 7861 must be free for the localhost phase.

## Remaining limits

One generated study was used in each positive phase above. Subsequent real-study checks now cover [three Windows/Linux replays and eight Linux API requests](MULTI_STUDY_VERIFICATION.md), including cgroup peak-memory measurement. The [initial one-case record](LINUX_REAL_STUDY_BENCHMARK.md) is retained. Wider scanner/protocol and real compressed-study coverage remain outstanding. No public inference host is configured.
