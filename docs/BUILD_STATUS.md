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
- Original real-study Kaggle raw/cache parity records are retained in ignored model assets. Local Windows/Linux replay now covers three studies; wider scanner/protocol and real compressed-study coverage remain outstanding.
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

## Linux real-study replay and initial benchmark — 25 October 2026

- Replayed the existing authorized private study in the Linux image, with no container networking. All 159 DICOM hashes, input pixels and the 32/40 validity mask matched the original Kaggle reference/cache. All 12 direct/API scores were within 1.20e-7 of Kaggle, below the unchanged 1e-5 tolerance.
- Four API requests on that same study gave identical scores. First worker processing took 6.951 seconds; three subsequent requests had a 6.127-second median (6.090–6.253 seconds). This excludes upload, queue wait and model startup. Local multipart upload-to-result observations were 7.490–8.681 seconds.
- Measured the API container's lifetime cgroup-v2 memory peak: 1,022,607,360 bytes (975.234 MiB), including server, worker, HTTP probe and attributed cache/tmpfs. This excludes the separate direct verifier, Docker VM, browser and host. The container used a two-CPU quota, 2 GiB memory limit and no swap; no OOM or memory-max events occurred.
- Verified all 32 preview endpoints, token protection and deletion after each request. Both temporary containers were removed. Weights and the private input folder were mounted read-only; the model/application were unchanged. No GPU, payment, data download or public upload was used.
- This initial check covered one unique study; repeated requests did not increase study coverage. See [LINUX_REAL_STUDY_BENCHMARK.md](LINUX_REAL_STUDY_BENCHMARK.md) and the subsequent extension below.

## Three-study extension and portfolio record — 25 October 2026

- A private copy of the saved Kaggle export reran the unchanged three-case raw/cache gate and exported the other two cases. It completed successfully in 166.2 seconds on CPU, with internet off. Only originally selected input series were exported; their pixels/masks still matched the original cache exactly. No GPU, training or competition submission was started.
- Imported two private exports into ignored `data/additional_replays/`. Both passed direct and actual-API verification on Windows and in network-isolated Linux containers. All three cases now cover 464 provided DICOM files, 32/40/32 valid windows and 512/640/672/1024-square matrices. Maximum score difference from Kaggle across both platforms: 1.1920928955078125e-7, below the unchanged 1e-5 tolerance.
- Eight total Linux API requests produced identical repeated score vectors within each case. Worker processing observations ranged from 3.871 to 10.140 seconds; maximum API-container cgroup peak was 1,208,995,840 bytes (1152.988 MiB), under the existing two-CPU/2-GiB configuration. No OOM or memory-max events occurred. Detailed timings and measurement scope are in [MULTI_STUDY_VERIFICATION.md](MULTI_STUDY_VERIFICATION.md).
- The 203-slice case included all five slots and 120,061,952 source pixels, exercising the existing 128-million-pixel admission limit. No upload/pixel limits were raised. The other new case retained the missing-coronal-non-FS mask.
- All three studies use Explicit VR Little Endian. Decoder availability is tested separately; no broader real compressed-DICOM coverage is claimed.
- Cleared API jobs, stopped the temporary Windows server and removed all temporary replay containers. Private data, references and screenshots remain excluded from Git. Added current portfolio/resume wording with label/pretraining attribution and a separate identity for the 0.950 public-model reproduction.
