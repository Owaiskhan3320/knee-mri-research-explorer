"""Append to a PRIVATE copy of the saved CPU raw-parity notebook.

Export the other two fixed parity cases, retaining only the input series selected
by the original pipeline. Outputs contain restricted MRI data: keep them private.
No training, GPU, competition submission, external requests or paid services.
"""
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import pydicom
import torch
import raw_preprocess as raw

assert not torch.cuda.is_available(), "Use accelerator None."
assert summary["complete"] and summary["gpu_used"] is False
assert len(summary["parity_studies"]) == 3
root = runner.competition_root()
all_metadata = pd.read_csv(root / "train_series.csv", dtype=str)
columns = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"]
model, cfg, _ = runner.load_model(OUT / "bundle", torch.device("cpu"))
exports = []
for case_number, reference_study in enumerate(summary["parity_studies"][1:], start=2):
    study_uid = reference_study["StudyInstanceUID"]
    selected_series = {d["series"] for d in reference_study["details"] if not d.get("missing")}
    original_metadata = all_metadata.loc[all_metadata.StudyInstanceUID.eq(study_uid), columns]
    metadata = original_metadata.loc[original_metadata.SeriesInstanceUID.isin(selected_series)].copy()
    assert set(metadata.SeriesInstanceUID) == selected_series
    export_dir = OUT / "additional_replays" / f"case_{case_number:02d}"
    export_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(export_dir / "series_metadata.csv", index=False)
    images, valid, details = raw.load_study(study_uid, raw.group_series(metadata), root / "train_series")
    image_hash = hashlib.sha256(images.tobytes() + valid.tobytes()).hexdigest()
    assert image_hash == reference_study["image_hash"], "Selected-series export changed original pixels/mask."
    assert not any(d.get("errors") or d.get("fallback_errors") for d in details)
    scores = runner.probabilities(model, images, valid, torch.device("cpu"))
    np.savez_compressed(export_dir / "reference_inputs.npz", images=images, valid=valid)
    files = []
    file_hashes, shapes, codecs = {}, {}, {}
    expanded_bytes, pixels = 0, 0
    for series_uid in metadata.SeriesInstanceUID:
        series_files = sorted((root / "train_series" / study_uid / series_uid).glob("*.dcm"))
        assert series_files
        for path in series_files:
            assert path.stat().st_size <= 16 * 1024**2
            expanded_bytes += path.stat().st_size
            header = pydicom.dcmread(path, stop_before_pixels=True)
            assert int(getattr(header, "NumberOfFrames", 1)) == 1
            height, width = int(header.Rows), int(header.Columns)
            pixels += height * width
            shape = f"{height}x{width}"
            codec = str(header.file_meta.TransferSyntaxUID)
            shapes[shape] = shapes.get(shape, 0) + 1
            codecs[codec] = codecs.get(codec, 0) + 1
            relative = path.relative_to(root / "train_series").as_posix()
            file_hashes[relative] = runner.sha256(path)
            files.append((path, relative))
    assert len(files) <= 4000 and expanded_bytes <= 512 * 1024**2 and pixels <= 128_000_000
    with zipfile.ZipFile(export_dir / "study.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path, relative in files:
            archive.write(path, relative)
    archive_bytes = (export_dir / "study.zip").stat().st_size
    assert archive_bytes <= 150 * 1024**2, "Case exceeds the app's existing upload limit; do not raise it here."
    profile = {"selected_series_only": True, "original_metadata_series_count": len(original_metadata),
               "exported_series_count": len(metadata), "matrix_shape_counts": shapes,
               "transfer_syntax_counts": codecs, "total_pixels": pixels,
               "archive_bytes": archive_bytes, "expanded_bytes": expanded_bytes}
    report = {
        "schema_version": 1, "source": "private Kaggle raw/cache parity replay",
        "parent_notebook_version": 360713817, "study_uid": study_uid,
        "gpu_used": False, "training_started": False, "checkpoint_sha256": runner.DEPLOY_SHA,
        "raw_preprocess_sha256": runner.sha256(OUT / "bundle" / "raw_preprocess.py"),
        "pixel_key": cfg["pixel_key"], "image_and_mask_sha256": image_hash,
        "valid_windows": np.flatnonzero(valid).tolist(), "scores": dict(zip(runner.TARGETS, map(float, scores))),
        "environment": runner.environment(), "source_file_count": len(files), "file_sha256": file_hashes,
        "archive_sha256": runner.sha256(export_dir / "study.zip"),
        "metadata_sha256": runner.sha256(export_dir / "series_metadata.csv"),
        "reference_inputs_sha256": runner.sha256(export_dir / "reference_inputs.npz"),
        "local_score_absolute_tolerance": 1e-5, "input_profile": profile,
        "notice": "Private research verification only. Selected input series only; identical original cache pixels/mask. No clinical validation. Do not redistribute scans."
    }
    runner.write_json(export_dir / "reference.json", report)
    (export_dir / "PRIVATE_DATA_NOTICE.txt").write_text(report["notice"])
    exports.append({"case": case_number, "files": len(files), "windows": int(valid.sum()), "input_profile": profile})
    print("PRIVATE_ADDITIONAL_REPLAY_EXPORTED", json.dumps(exports[-1]), flush=True)
runner.write_json(OUT / "additional_replays" / "export_summary.json", {"complete": True, "gpu_used": False, "cases": exports})
print("PRIVATE_ADDITIONAL_REPLAYS_COMPLETE", flush=True)
