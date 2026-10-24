"""Append to the existing private raw-parity notebook; CPU only.

Exports ONE already authorized competition study for private local replay.
Keep the notebook and its raw-data outputs private. Never commit the outputs.
The unchanged parent notebook runs its original raw/cache parity gate first.
"""
import csv
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import raw_preprocess as raw

assert not torch.cuda.is_available(), "Use accelerator None."
assert summary["complete"] and summary["gpu_used"] is False
reference_study = summary["parity_studies"][0]
study_uid = reference_study["StudyInstanceUID"]
root = runner.competition_root()
metadata = pd.read_csv(root / "train_series.csv", dtype=str)
columns = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"]
metadata = metadata.loc[metadata.StudyInstanceUID.eq(study_uid), columns].copy()
assert 1 <= len(metadata) <= 100
export_dir = OUT / "local_replay"
export_dir.mkdir(exist_ok=True)
metadata.to_csv(export_dir / "series_metadata.csv", index=False)
images, valid, details = raw.load_study(study_uid, raw.group_series(metadata), root / "train_series")
image_hash = hashlib.sha256(images.tobytes() + valid.tobytes()).hexdigest()
assert image_hash == reference_study["image_hash"], "Original pixel parity must still hold."
model, cfg, _ = runner.load_model(OUT / "bundle", torch.device("cpu"))
scores = runner.probabilities(model, images, valid, torch.device("cpu"))
np.savez_compressed(export_dir / "reference_inputs.npz", images=images, valid=valid)
file_hashes = {}
with zipfile.ZipFile(export_dir / "study.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for series_uid in metadata.SeriesInstanceUID:
        files = sorted((root / "train_series" / study_uid / series_uid).glob("*.dcm"))
        assert files, "Every metadata series must have files."
        for path in files:
            relative = path.relative_to(root / "train_series").as_posix()
            file_hashes[relative] = runner.sha256(path)
            archive.write(path, relative)
assert (export_dir / "study.zip").stat().st_size < 150 * 1024**2
report = {
    "schema_version": 1, "source": "private Kaggle raw/cache parity replay",
    "parent_notebook_version": 360713817,
    "study_uid": study_uid, "gpu_used": False, "training_started": False,
    "checkpoint_sha256": runner.DEPLOY_SHA,
    "raw_preprocess_sha256": runner.sha256(OUT / "bundle" / "raw_preprocess.py"),
    "pixel_key": cfg["pixel_key"], "image_and_mask_sha256": image_hash,
    "valid_windows": np.flatnonzero(valid).tolist(),
    "scores": dict(zip(runner.TARGETS, map(float, scores))),
    "environment": runner.environment(), "source_file_count": len(file_hashes),
    "file_sha256": file_hashes,
    "archive_sha256": runner.sha256(export_dir / "study.zip"),
    "metadata_sha256": runner.sha256(export_dir / "series_metadata.csv"),
    "reference_inputs_sha256": runner.sha256(export_dir / "reference_inputs.npz"),
    "local_score_absolute_tolerance": 1e-5,
    "notice": "Private research verification only. This is not accuracy or clinical validation. Do not redistribute scans."
}
runner.write_json(export_dir / "reference.json", report)
(export_dir / "PRIVATE_DATA_NOTICE.txt").write_text(report["notice"])
print("PRIVATE_LOCAL_REPLAY_EXPORT_COMPLETE", json.dumps({"files": len(file_hashes), "windows": int(valid.sum()), "archive_MB": round((export_dir / "study.zip").stat().st_size / 1024**2, 2), "gpu_used": False}))
