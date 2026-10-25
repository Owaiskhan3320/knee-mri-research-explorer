# Local MVP verification — 24 October 2026

The application was verified in a dedicated Windows Python3.12 environment with CPU-only torch2.11.0+cpu and torchvision0.26.0+cpu. All resolved application packages are recorded in requirements-lock.txt. pip check reports no broken requirements.

## Verified

- Downloaded the already saved private Kaggle raw-parity output; no Kaggle training, GPU session or new submission was started.
- Imported the exact112,422,539-byte epoch8 deploy.pt checkpoint, SHA256 e30535be63503f489d3e0961afa51cce83bfe7d9766faffbe5dfc2be454310dd.
- Copied the original raw_preprocess.py byte for byte. Extracted the unchanged StudyMIL class without its training-runtime imports. Startup verifies all model/preprocessing identities and strict-loads the weights.
- **16 automated tests passed in44.99seconds.** Includes compressed-DICOM decoder availability, actual model inference on generated DICOMs, masking invariance, request independence, archive traversal/expansion/pixel limits, metadata consistency, queue bounds, job access tokens, expiry, upload disabling, and forced timeout followed by worker restart and successful inference.
- Loopback API end-to-end test passed: validation, preprocessing, inference,40valid windows,12finite scores, protected preview/result access and deletion. Generated-fixture timing:5.453seconds total,4.078seconds model inference. This is not a real-study benchmark.
- Multipart upload of the same generated fixture produced exactly the same scores as the built-in path.
- Real-study Windows replay passed: 159 DICOMs, 32/40 valid windows, exact source hashes/pixels/mask, all 12 scores within 5.96e-8 of Kaggle CPU inference. Direct local and API scores were identical. API worker processing took 9.078 seconds for this one request. Raw data and detailed results are ignored under data/local_replay/; see REAL_STUDY_VERIFICATION.md.
- Browser interaction passed: actual model run, completed scores, decoded224x224preview, sampled-window control. A subpath image-routing issue was fixed by mounting the interface at the root URL.
- Local Git initialized with no remote. Verified .gitignore excludes model weights, .venv and runtime studies/results.

## Limits

- Synthetic fixtures are mathematical phantoms, not patient scans, and do not establish accuracy or clinical utility.
- Original real-study Kaggle raw/cache parity records are retained in ignored model assets. Windows replay now covers one real study; a broader scanner/codec/performance benchmark remains outstanding.
- Docker build and runtime checks passed on 25 October; see the container checks below. The local Python app also works without Docker.
- No public inference hosting, payment, subscription, paid API or cloud inference is configured or used. Source publication is documented in the 25 October release checks below.
- No real competition scans are bundled for public display. Arbitrary clinical-series metadata inference is outside this first version.

## Source release checks — 25 October 2026

- The full suite now has 18 tests, including two checks that a rejected model bundle leaves existing assets intact. All 18 passed in the primary environment (91.35 seconds).
- A new isolated source checkout and virtual environment were created. Setup found a Python-launcher discovery issue, which was fixed by detecting installed Python 3.12 or accepting an explicit executable path.
- Fresh installation from pinned packages succeeded, and pip check passed. Without private weights: 15 passed, 3 explicitly skipped. After verified bundle import: all 18 passed (85.80 seconds). Import did not modify application source.
- Fixed a narrow-screen flex layout overflow. Browser verification showed page/root width within the available viewport when result details were expanded.
- A 60-second captioned MP4 was created locally from nine actual generated-data browser captures. Frames are held for readability; there is no narration or real-time latency claim. Real scans are not in the video or public screenshot.
- Pilkwang Kim's report-label source, pretrained initialization, learning-method references, software dependencies and AI coding assistance are credited in THIRD_PARTY.md.
- A source/media allowlist audit checks Git's file set for excluded assets, likely credentials and real study identifiers before pushing.
- Added .gitattributes to preserve LF source bytes on Windows/Linux, because model/preprocessing startup checks use exact file hashes. A real clone from the public GitHub repository reproduced both frozen source hashes, passed 15 tests with 3 explicit model skips (17.17 seconds), and imported/strict-loaded the authorized model locally without modifying source.

Source repository: https://github.com/Owaiskhan3320/knee-mri-research-explorer. No live inference host or paid service is configured.

## Local container checks — 25 October 2026

- Recovered Docker Desktop startup by preserving and recreating two stale transient socket directories. No factory reset, image/volume removal or WSL distribution reset was used.
- Built a Linux/amd64 Python 3.12 CPU image on the local Docker Desktop engine. Build-time `pip check`, exact CPU torch/torchvision version checks and JPEG Lossless/JPEG 2000 decoder availability passed.
- Restricted the Docker context to application Python source and dependency manifests. No weights, real scans, runtime files or credentials are baked into the image. All application source hashes match the host checkout.
- Three runtime phases passed: missing weights (readiness 503 and inference request rejected), networking-disabled actual-model inference, and loopback-only host access. Positive phases used a non-root user, a read-only root filesystem/model mount, two CPUs and a 2 GiB memory limit. Token-protected access, cleanup, health checks and disabled-upload policy passed.
- The input was a generated DICOM phantom. These checks establish container mechanics, not Linux real-study parity or medical accuracy. Recorded memory values are post-request snapshots, not peak-memory measurements.
- No GPU, remote builder, image push or paid service was used. Temporary test containers were removed; the local image and private model files were retained. See [CONTAINER_VERIFICATION.md](CONTAINER_VERIFICATION.md) and its machine-readable report.

Open http://127.0.0.1:7860/ while the server is running. Restart with start.ps1 or `.venv\Scripts\python.exe run.py` from the project folder. Only loopback is bound by default.
