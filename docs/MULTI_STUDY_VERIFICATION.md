# Three-study Windows/Linux deployment check — 25 October 2026

**Passed: three unique authorized research studies, 464 provided DICOM files.** Every source-file hash, processed pixel and validity mask matched the private Kaggle reference on Windows and Linux. All 12 scores per case were within **1.1920928955078125e-7** of Kaggle CPU inference, below the original **1e-5** tolerance. These checks test implementation consistency, without evaluating medical accuracy.

The model remains the project-trained ConvNeXt-Tiny epoch 8 with **0.833 public macro ROC AUC**. The checkpoint, scientific preprocessing, architecture and inference image were unchanged.

## Cases and input scope

| Case | Provided DICOMs | Matrix dimensions | Valid windows | Input condition |
| --- | ---: | --- | ---: | --- |
| 01, original | 159 | 512², 640², 672² | 32/40 | Missing coronal non-fat-suppressed slot |
| 02, additional | 203 | 512², 1024² | 40/40 | All five slots present |
| 03, additional | 102 | 512², 672² | 32/40 | Missing coronal non-fat-suppressed slot |

Cases 02/03 were exported from a private copy of the original raw/cache parity notebook using only the series selected by that original pipeline. Unused candidate series were excluded before export; reprocessing the selected series still produced the exact original cached arrays and masks. The figures above count provided files, rather than every sequence in the original full studies. Case 01 retains its original export.

The private [additional export, version 1](https://www.kaggle.com/code/owaiskhan15/knee-additional-local-replay-export?scriptVersionId=362222429) preserved the original parity gate and completed successfully in **166.2 seconds**, with accelerator None and internet off. No training, new leaderboard submission or paid service was used. The original export notebook/version was preserved. Export source is in [kaggle_export_additional_replays.py](../scripts/kaggle_export_additional_replays.py).

Case 02 contained **120,061,952 source pixels**, below the existing 128-million-pixel admission limit. Its ZIP was 112,655,391 bytes, below the existing 150 MiB upload limit. No limits were raised to admit these cases. All three used **Explicit VR Little Endian**, an uncompressed transfer syntax. Real compressed-study and wider scanner/protocol coverage remain open.

## Consistency and API results

| Case | Maximum Windows/Kaggle score difference | Maximum Linux/Kaggle score difference | Linux API requests |
| --- | ---: | ---: | ---: |
| 01 | 5.96e-8 | 1.19e-7 | 4 |
| 02 | 5.96e-8 | 4.47e-8 | 2 |
| 03 | 7.45e-8 | 1.19e-7 | 2 |

Direct and API paths passed on each platform. All repeated Linux requests gave identical score vectors within each case. Every valid preview endpoint was checked, job access required its token, and jobs were cleared after verification. Eight Linux API requests represent three unique studies. Temporary Linux containers were removed and the temporary Windows server was stopped; private inputs were retained locally.

## Observed resource use

Host: Intel Core i5-1235U, 10 physical cores/12 logical processors, approximately 16 GB RAM, Windows with Docker Desktop/WSL2. The Linux containers had a **two-CPU quota, 2 GiB memory limit, no swap, no network**, read-only model/input mounts and a read-only root filesystem. Other host workloads were not stopped.

| Case | First Linux worker total | Subsequent Linux worker totals | API-container memory peak |
| --- | ---: | --- | ---: |
| 01 | 6.951 s | 6.090 / 6.253 / 6.127 s | 975.234 MiB |
| 02 | 10.140 s | 9.244 s | 1152.988 MiB |
| 03 | 5.103 s | 3.871 s | 836.129 MiB |

The largest cgroup-accounted peak was **1,208,995,840 bytes (about 1.13 GiB)**. It includes the API server, model worker, HTTP probe and attributed cache/tmpfs, and excludes the separate direct verifier, Docker/WSL VM overhead, Windows host and browser. No memory-max or OOM events occurred. The fixed resource limits passed these inputs; minimum hosting capacity and concurrent-load behavior were not measured.

Worker timings include validation, preprocessing, inference and preview generation; they exclude upload, queue wait and model startup. The model was ready before each first request. Single Windows API worker observations were 9.078 / 33.672 / 20.188 seconds for cases 01/02/03. These measurements occurred at different times on a shared host, so they do not isolate operating-system performance. Per-request local upload-to-result observations and exact counters are in [multi_study_report.json](multi_study_report.json).

## Repeat with authorized private assets

The original study remains in ignored `data/local_replay/`; the two new exports are in `data/additional_replays/case_02/` and `case_03/`. Each contains `study.zip`, `series_metadata.csv`, `reference_inputs.npz`, `reference.json` and a private-data notice. None is published.

Start the local app and run the Windows verifier against each folder, using the appropriate loopback port:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\verify_real_study.py' 'data\additional_replays\case_02' --base-url http://127.0.0.1:7860
```

For Linux, with the existing image and Docker engine ready:

```powershell
& '.\.venv\Scripts\python.exe' 'scripts\benchmark_container.py' 'data\additional_replays\case_02' --warm-repeats 1 --output 'data\additional_replays\case_02\linux_real_study_report.json'
```

Repeat for case 03. The Linux script enables uploads only inside its network-isolated test container; the image's default upload policy remains off. The original [Windows baseline](REAL_STUDY_VERIFICATION.md) and [four-request Linux benchmark](LINUX_REAL_STUDY_BENCHMARK.md) remain available.

This small selected sample does not establish clinical utility, calibration, held-out accuracy, site/scanner generalization, tail latency or a public-hosting specification. The portfolio records the verified engineering result and preserves the model's original evaluation boundaries.
