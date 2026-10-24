# Real-study Windows verification — 24 October 2026

**Passed on one authorized research MRI study.** This is an implementation consistency check, not a clinical accuracy study.

The private CPU export notebook [Knee Local Windows Replay Export, version 1](https://www.kaggle.com/code/owaiskhan15/knee-local-windows-replay-export?scriptVersionId=361757109) is a copy of the original raw/cache parity notebook, version 360713817. The original notebook was preserved. The copy reran the original three-study parity gate and exported the first fixed weak-validation study. It remained private, used no accelerator and completed in 1 minute 41 seconds. No training or competition submission was started.

| Check | Result |
| --- | --- |
| Raw files | 159 DICOM slices; every SHA256 matched |
| Processed model input | Exact equality of all 40 × 3 × 224 × 224 uint8 pixels |
| Validity mask | Exact equality; 32 valid windows out of 40 |
| Original cache identity | Pixel/mask hash matched the original saved Kaggle parity record |
| Findings compared | All 12, by exact target name |
| Score tolerance | Absolute difference ≤ 0.00001, fixed before local evaluation |
| Maximum direct Windows/Kaggle score difference | 0.00000005960464477539063 |
| Maximum local API/Kaggle score difference | 0.00000005960464477539063 |
| Direct Windows/local API score difference | 0.0 |
| Preview endpoints | All 32 valid-window PNGs returned successfully |
| Job access and cleanup | Token required; API verification job removed after comparison |

One coronal non-fat-suppressed sequence was unavailable. Its eight windows remained masked; no images or predictions were invented to fill them.

The checkpoint remained the project-trained ConvNeXt-Tiny epoch 8, SHA256 `e30535be63503f489d3e0961afa51cce83bfe7d9766faffbe5dfc2be454310dd`. Frozen preprocessing source was unchanged. Both environments used torch 2.11.0 CPU, torchvision 0.26.0 CPU, numpy 2.1.3, Pillow 12.3.0 and pydicom 3.0.2; platform-specific decoder wheels used matching package versions.

## One-request measurements

The local API worker reported 9.078 seconds total processing, including 1.078 seconds preprocessing and 2.703 seconds model inference. Total includes archive/header validation and preview generation; it excludes browser upload/network time and initial server/model startup. A subsequent browser run displayed approximately 9.0 seconds.

The separate verifier process peaked at 644.1 MB working set, including its own model load and inference. This is **not** aggregate memory for the browser, API and persistent model worker. These observations are not median or tail-latency benchmarks and do not establish minimum hosting specifications.

## Reproduce privately

Start the local app. Put the private export's five files under `data/local_replay/`, then run:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\verify_real_study.py' 'data\local_replay'
```

The verifier admits only a loopback HTTP endpoint, verifies artifact/source hashes, checks exact input arrays, runs local CPU inference and tests the actual multipart API. It writes the detailed comparison and result under the ignored data directory, then clears the API job. Export/reference files are retained locally for repeatability.

Do not publish `study.zip`, DICOMs, reference input arrays, real-study screenshots or detailed study identifiers. Raw data are not bundled with the public code. The generated software phantom remains the shareable demo input.

This one case does not establish scanner/site generalization, diagnostic reliability, calibration, clinical utility or broad codec coverage. The public competition macro ROC AUC remains **0.833**, not a newly measured score.
