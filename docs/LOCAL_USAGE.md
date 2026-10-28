# Local setup and operation

This guide supports the [project overview](../README.md). The application is a local CPU research demonstration, not a diagnostic service. No payment, paid API, public inference host or Gradio public share link is configured. The full competition dataset is not required.

## Start on Windows

Python 3.12 is required. From the repository folder:

```powershell
.\setup.ps1
```

Setup creates an isolated `.venv/` and installs pinned CPU dependencies from free package registries. If Python 3.12 is outside PATH and the usual Windows install folder, pass its executable explicitly:

```powershell
.\setup.ps1 -PythonPath 'C:\path\to\python.exe'
```

**A public clone contains no trained weights or MRI data.** Inference requires an already downloaded bundle that you are authorized to access. See [model identity and bundle source](../models/README.md). Import it before starting the app:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\import_bundle.py' 'C:\path\to\authorized-output.zip'
.\start.ps1
```

Import verifies source/checkpoint identities before writing private model assets. It does not download weights or replace application source. If scripts are blocked by PowerShell policy and the environment is already installed, run the existing environment directly:

```powershell
& '.\.venv\Scripts\python.exe' 'run.py'
```

Open [the local interface](http://127.0.0.1:7860/) or [API documentation](http://127.0.0.1:7860/docs). The readiness endpoint returns 503 while the model loads or if weights are missing. Without authorized weights, inspect the code, watch the [generated-data demo](demo.mp4) or run the model-independent tests below; inference remains unavailable.

Choose **Generated software test**, then **Run local analysis**. This creates an artificial DICOM phantom, not a knee scan. Its scores have no medical meaning. No real example scans are redistributed.

## Analyze an authorized local study

Choose **My study (local)**. Provide one ZIP containing `StudyUID/SeriesUID/*.dcm` and a CSV with these columns:

```text
StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fat_Suppression,Fluid_Sensitive
```

Plane values are `Sagittal`, `Coronal` or `Axial`. Flags are `0`/`1` or `true`/`false`. Study and series identifiers must match the archive. Unsupported or multiframe studies fail. Do not use identifiable patient data in portfolio demonstrations.

The viewer shows centre slices of sampled, cropped model windows. Results contain 12 uncalibrated scores, validity/warning information, stage timings and checkpoint provenance. They are not calibrated risks, diagnostic decisions or lesion maps. Use **Download result JSON** to save a local result and **Clear study** to remove its application-held data.

## API and data handling

`POST /api/v1/jobs` with multipart `kind=synthetic` returns a job ID and access token. For local uploads, use `kind=upload`, `study` and `metadata` files. Pass `Authorization: Bearer TOKEN` to job status, result, preview and delete endpoints. Poll status and retrieve the result only after completion. Keep tokens private.

The app processes one study at a time with three waiting jobs. Study ZIPs, extracted data, previews and results live under ignored `.runtime/jobs/`. They expire 30 minutes after completion and are removed on error, clear or shutdown. Stale job directories are cleaned at next startup. The interface has a separate `.runtime/gradio/` file cache with approximately 30-minute expiry and shutdown cleanup. Downloaded copies are outside application cleanup.

Keep the app local. The input checks, job tokens and cleanup do not constitute a clinical deployment or a complete public-service security design. [Architecture](ARCHITECTURE.md) describes the queue, deadlines, worker restart and input limits. `.gitignore` excludes private weights, studies, runtime files and credentials.

## Tests and repeat checks

From the repository root, using the installed environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Three model-dependent application tests are explicitly skipped when weights are absent. Require them with:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q --require-model
```

The second command fails if private weights are missing or invalid; it does not substitute a random model. The historical application checks passed on a fresh Windows Python 3.12 environment on 25 October 2026: **15 passed, 3 skipped** without weights and **18 passed** after verified import. Those counts predate the later evaluator/analysis tests. [Verification history](BUILD_STATUS.md) records their scope.

The [28 October final release review](RELEASE_NOTES.md) ran the expanded suite: **70 passed** with authorized weights and **67 passed, 3 skipped** in a clean public source clone without weights/data. Both used the existing installed environment; the source-clone check was not a new dependency installation.

The training archive and planned pooling protocol can be inspected with standard-library-only checks, without weights, MRI data or training:

```powershell
& '.\.venv\Scripts\python.exe' -S 'scripts\verify_training_archive.py'
& '.\.venv\Scripts\python.exe' -S 'scripts\verify_pooling_protocol.py'
```

The protocol check does not run or authorize the deferred experiment. To reproduce historical numerical results with authorized private saved outputs, follow [per-finding evaluation](PER_FINDING_EVALUATION.md), [training history](TRAINING_HISTORY.md) and [error analysis](ERROR_ANALYSIS.md).

To repeat deployment parity with your own authorized export, start the app first, then run:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\verify_real_study.py' 'data\local_replay'
```

Private exports are optional and excluded from Git/Docker. The [three-study record](MULTI_STUDY_VERIFICATION.md) explains exact input and score comparisons, CPU timings and memory scope. The [portfolio checklist](PORTFOLIO.md) describes publishable evidence and attribution.

## Docker (optional, local CPU)

The image was built and tested on Windows/WSL2 with a Linux engine on 25 October 2026. Import authorized weights first and start Docker Desktop's Linux engine. From the project folder:

```powershell
docker --context desktop-linux build --builder desktop-linux -t knee-mri-explorer:local .
docker --context desktop-linux run --rm --init --name knee-mri-explorer --read-only --cpus 2 --memory 2g --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,noexec,size=128m --tmpfs /opt/knee/.runtime:rw,nosuid,noexec,size=512m,uid=1000,gid=1000 -p 127.0.0.1:7861:7860 --mount "type=bind,source=$PWD\models,target=/opt/knee/models,readonly" knee-mri-explorer:local
```

Open [the container interface](http://127.0.0.1:7861/) and choose **Generated software test**. The process runs as a non-root user, mounts private weights read-only and keeps temporary files in memory-backed directories. Uploads default off. Stop with Ctrl+C; the image and model files remain local. A cold build downloads dependencies; subsequent offline inference checks do not require network access.

After building, the automated check runs missing-model, networking-disabled inference and localhost-access phases, then removes its temporary containers:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\verify_container.py'
```

See [container checks](CONTAINER_VERIFICATION.md) and [three-study replay/performance](MULTI_STUDY_VERIFICATION.md) for results and limits. Wider scanner/protocol and real compressed-study coverage remain unverified. No image is pushed and no paid hosting is used.
