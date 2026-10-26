"""CPU preparation run inside Kaggle; does not train or submit a model."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(output):
    assert not torch.cuda.is_available(), "CPU preparation requires accelerator None"
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    started = time.monotonic()
    output = Path(output)
    bundle = output / "bundle"
    audit = output / "label_audit"
    images_out = output / "image_audit"
    for folder in [output, bundle, audit, images_out]:
        folder.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation_status.json", {"complete": False, "gpu_used": False})
    root = Path("/kaggle/input")
    prep = root / "notebooks/owaiskhan15/knee-dinov2-small-mil/knee_dino_small"
    old_dino = root / "notebooks/owaiskhan15/knee-dino-cpu-head-extended/dino_mil_head_extended"
    feature = root / "notebooks/owaiskhan15/knee-dino-frozen-features/dino_frozen_features"
    baseline = root / "notebooks/owaiskhan15/notebookb72222f4d7/knee_baseline"
    prior_audit = root / "notebooks/owaiskhan15/knee-cpu-audit-verified/knee_cpu_audit"
    competition = root / "competitions/rsna-knee-abnormality-detection"
    pilkwang = root / "datasets/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv"
    steven = root / "datasets/stevenleehans/rsna-knee-llm-report-labels"
    jev = root / "datasets/riadmohamed42/jev-knee-labels"
    pixel_roots = [root / "notebooks/owaiskhan15" / slug / "knee_dino_small" for slug in
                   ["knee-dino-cpu-pixel-cache", "knee-dino-cpu-pixel-cache-b"]]
    for path in [prep / "study_labels_splits.csv", old_dino / "dino_mil_deploy.pt",
                 prior_audit / "run_summary.json", pilkwang, steven, jev]:
        assert path.exists(), f"Required attached input missing: {path}"
    dino = torch.load(old_dino / "dino_mil_deploy.pt", map_location="cpu", weights_only=True)
    targets = dino["targets"]
    slots = dino["config"]["slots"]
    assert len(targets) == 12 and len(slots) == 5
    pixel_key = dino["feature_spec"]["pixel_key"]
    data_hashes = json.loads((prep / "data_hashes.json").read_text())
    assert data_hashes == dino["data_hashes"]
    for key, path in [("labels", pilkwang), ("original_split", baseline / "splits.csv"),
                      ("patient_manifest", baseline / "cache_manifest.csv"),
                      ("train_csv", competition / "train.csv"),
                      ("series_csv", competition / "train_series.csv")]:
        assert sha256(path) == data_hashes[key], f"Original input changed: {key}"
    old_summary = json.loads((prior_audit / "run_summary.json").read_text())
    assert old_summary["complete"] and old_summary["validation_studies"] == 637
    frames = pd.read_csv(prep / "study_labels_splits.csv", dtype={"StudyInstanceUID": str})
    assert frames.groupby("split").size().to_dict() == {"fit": 3655, "val": 637, "gold": 58}
    assert frames.StudyInstanceUID.is_unique
    assert frames.groupby("verified_group").split.nunique().max() == 1
    labels = pd.read_csv(pilkwang, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    assert labels.index.is_unique
    aligned_labels = labels.reindex(frames.StudyInstanceUID)
    frames["split"] = frames["split"].replace({"val": "weak_validation"})
    for target in targets:
        verdict = aligned_labels[target + "__verdict"].astype(str).str.upper().str.strip()
        raw_soft = pd.to_numeric(aligned_labels[target], errors="coerce").to_numpy(float)
        known = verdict.isin(["YES", "NO"]).to_numpy() & np.isfinite(raw_soft)
        weak_rows = frames.split.ne("gold").to_numpy()
        expected_soft = np.where(known, raw_soft, np.nan)
        assert np.allclose(frames.loc[weak_rows, target], expected_soft[weak_rows], equal_nan=True)
        binary = np.where(known, verdict.eq("YES").to_numpy().astype(float), np.nan)
        binary[~weak_rows] = frames.loc[~weak_rows, target].to_numpy(float)
        frames[target + "__binary"] = binary
    columns = ["StudyInstanceUID", "split", "verified_group"] + targets + [t + "__binary" for t in targets]
    frames = frames[columns].copy()
    frames.to_csv(bundle / "study_labels_splits.csv", index=False)
    print("Verified original split and unchanged Pilkwang training targets:", frames.groupby("split").size().to_dict(), flush=True)

    # Compare alternatives without changing the training labels or selecting on expert-58.
    sys.path.insert(0, str(bundle))
    from label_audit import run_label_audit
    label_report = run_label_audit(pilkwang, steven, jev, frames, audit, targets)
    assert label_report["optional_sources_complete"], "Expected comparison label files are missing"
    print("Label audit completed; training source remains Pilkwang.", flush=True)

    parts = []
    partition_provenance = []
    for cache_root in pixel_roots:
        status = json.loads((cache_root / "pixel_status.json").read_text())
        assert status["complete"] and status["key"] == pixel_key
        part = pd.read_csv(cache_root / "pixel_manifest.csv", dtype={"StudyInstanceUID": str})
        for column in ["StudyInstanceUID", "shard", "member", "packed_sha256"]:
            assert column in part and part[column].notna().all()
        part["root"] = str(cache_root)
        parts.append(part[["StudyInstanceUID", "root", "shard", "member", "packed_sha256"]])
        partition_provenance.append({"root": str(cache_root), "manifest_sha256": sha256(cache_root / "pixel_manifest.csv"),
                                     "status_sha256": sha256(cache_root / "pixel_status.json"), "rows": len(part)})
    pixels = pd.concat(parts, ignore_index=True)
    assert pixels.StudyInstanceUID.is_unique and set(pixels.StudyInstanceUID) == set(frames.StudyInstanceUID)
    pixels = pixels.set_index("StudyInstanceUID").loc[frames.StudyInstanceUID].reset_index()
    pixels.to_csv(bundle / "pixel_index.csv", index=False)
    with np.load(feature / "frozen_features.npz", allow_pickle=False) as cached_features:
        feature_uids = cached_features["uids"].astype(str)
        valid_masks = cached_features["valid"].copy()
        assert str(cached_features["key"].item()) == dino["feature_key"]
        assert np.array_equal(cached_features["slots"], np.repeat(np.arange(5), 8))
        assert cached_features["completed"].all()
    positions = {uid: i for i, uid in enumerate(feature_uids)}
    aligned_masks = valid_masks[[positions[u] for u in frames.StudyInstanceUID]]
    assert aligned_masks.shape == (4350, 40) and aligned_masks.any(1).all()
    window_counts = aligned_masks.reshape(4350, 5, 8).sum(2)
    coverage = frames[["StudyInstanceUID", "split"]].copy()
    for i, (plane, fat_suppressed) in enumerate(slots):
        coverage[f"{plane}_{'FS' if fat_suppressed else 'nonFS'}"] = window_counts[:, i]
    coverage.to_csv(images_out / "sequence_coverage.csv", index=False)

    # A deterministic QC sample, not a medical annotation or a full anatomical audit.
    rng = np.random.default_rng(42)
    weak_indices = np.flatnonzero(frames.split.ne("gold"))
    sampled = set(rng.choice(weak_indices, 48, replace=False).tolist())
    for i in range(5):
        missing = weak_indices[window_counts[weak_indices, i] == 0]
        if len(missing):
            sampled.update(rng.choice(missing, min(3, len(missing)), replace=False).tolist())
    sampled = sorted(sampled)
    handles = {}
    qc_rows = []
    montage = []
    try:
        for number, index in enumerate(sampled):
            row = pixels.iloc[index]
            zip_path = Path(row.root) / "pixels" / row.shard
            if str(zip_path) not in handles:
                handles[str(zip_path)] = zipfile.ZipFile(zip_path)
            body = handles[str(zip_path)].read(row.member)
            assert hashlib.sha256(body).hexdigest() == row.packed_sha256
            with np.load(io.BytesIO(body), allow_pickle=False) as packed:
                uid = str(packed["uid"].item())
                assert uid == row.StudyInstanceUID and str(packed["key"].item()) == pixel_key
                images, refs, valid = packed["frames"], packed["refs"], packed["valid"]
                assert images.dtype == np.uint8 and images.ndim == 3 and images.shape[1:] == (224, 224)
                assert refs.shape == (40, 3) and valid.shape == (40,) and valid.dtype == bool
                assert refs.min() >= 0 and refs.max() < len(images)
                assert np.array_equal(valid, aligned_masks[index])
                dense = images[refs]
                assert np.std(dense[valid].astype(np.float32)) > 0
                qc_rows.append({"StudyInstanceUID": uid, "split": frames.iloc[index].split,
                                "valid_windows": int(valid.sum()), "unique_frames": len(images),
                                "zero_fraction": float((dense[valid] == 0).mean()),
                                "image_std": float(dense[valid].astype(np.float32).std())})
                if len(montage) < 12:
                    montage.append((uid, dense.copy(), valid.copy()))
    finally:
        for handle in handles.values():
            handle.close()
    pd.DataFrame(qc_rows).to_csv(images_out / "sample_integrity_checks.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(montage), 5, figsize=(11, 2 * len(montage)), squeeze=False)
    for row_number, (uid, dense, valid) in enumerate(montage):
        for slot_number, (plane, fs) in enumerate(slots):
            ax = axes[row_number, slot_number]
            available = np.flatnonzero(valid[slot_number * 8:(slot_number + 1) * 8])
            if len(available):
                ax.imshow(dense[slot_number * 8 + available[len(available) // 2], 1], cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(.5, .5, "Missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            ax.set_title(f"{plane} {'FS' if fs else 'non-FS'} | {len(available)}/8", fontsize=8)
            if slot_number == 0:
                ax.text(-.02, .5, "..." + uid[-10:], ha="right", va="center", transform=ax.transAxes, fontsize=7)
    fig.suptitle("Existing cached MRI inputs: technical QC sample, no medical labels", fontsize=12)
    fig.tight_layout(rect=[.06, 0, 1, .99])
    fig.savefig(images_out / "cached_input_qc.png", dpi=110)
    plt.close(fig)
    print("Pixel checks passed:", len(qc_rows), "sampled studies; all 4350 manifest entries aligned.", flush=True)

    from cnn_training import StudyMIL, cpu_preflight
    official_weights = EfficientNet_B0_Weights.DEFAULT
    print("Downloading official torchvision EfficientNet-B0 initialization on CPU:", official_weights.url, flush=True)
    official_encoder = efficientnet_b0(weights=official_weights)
    encoder_state = {k: v.detach().cpu() for k, v in official_encoder.state_dict().items()}
    model = StudyMIL(num_targets=12, slots=5, projection=256, encoder_state=encoder_state)
    assert all(p.requires_grad for p in model.encoder.parameters())
    torch.save(model.state_dict(), bundle / "initial_model.pt")
    reloaded = StudyMIL(num_targets=12, slots=5, projection=256)
    reloaded.load_state_dict(torch.load(bundle / "initial_model.pt", map_location="cpu", weights_only=True), strict=True)
    config = {"targets": targets, "seed": 42, "pixel_key": pixel_key, "feature_key": dino["feature_key"],
              "expected_counts": {"fit": 3655, "weak_validation": 637, "gold": 58},
              "slots": slots, "encoder": "torchvision.efficientnet_b0", "pretrained_weights": str(official_weights),
              "official_weight_url": official_weights.url, "selection": "minimum_full_weak_validation_soft_bce",
              "label_source": "Pilkwang report_labels_v2: YES/NO soft scores, UNK masked; unchanged",
              "data_hashes": data_hashes, "pixel_partitions": partition_provenance,
              "settings": {"batch_size": 2, "accumulation": 4, "epochs": 8, "workers": 2,
                           "encoder_lr": 3e-5, "head_lr": 3e-4, "weight_decay": 1e-4,
                           "train_windows_per_slot": 2, "validation_windows": 40, "projection": 256,
                           "patience": 3, "min_delta": 1e-4, "warmup_steps": 40,
                           "max_minutes": 90, "save_reserve_seconds": 20,
                           "intensity_jitter": .1, "freeze_bn_stats": True},
              "hashes": {name: sha256(bundle / name) for name in
                         ["study_labels_splits.csv", "pixel_index.csv", "initial_model.pt", "cnn_training.py", "label_audit.py", "prepare.py"]}}
    write_json(bundle / "config.json", config)
    preflight = cpu_preflight(bundle)
    assert preflight["complete"]
    write_json(bundle / "preflight.json", preflight)
    config["hashes"]["preflight.json"] = sha256(bundle / "preflight.json")
    write_json(bundle / "config.json", config)
    write_json(output / "source_provenance.json", {"data_hashes": data_hashes, "pixel_partitions": partition_provenance,
               "prior_audit_summary_sha256": sha256(prior_audit / "run_summary.json"),
               "encoder_weight_source": official_weights.url,
               "training_label_policy": "Unchanged Pilkwang soft labels; alternatives audited, not auto-selected",
               "expert_role": "Repeatedly used development diagnostic; not independent validation"})
    for file in prep.glob("*.json"):
        if "preprocess" in file.name:
            (images_out / ("original_" + file.name)).write_bytes(file.read_bytes())
    summary = {"complete": True, "gpu_used": False, "training_studies": 3655,
               "validation_studies": 637, "expert_development_studies": 58,
               "pixel_manifest_studies": len(pixels), "pixel_qc_sample_studies": len(qc_rows),
               "training_labels_changed": False, "cpu_preflight_passed": True,
               "planned_gpu_hard_cap_minutes": 90, "cpu_minutes": (time.monotonic() - started) / 60,
               "limitations": ["QC sample is not an exhaustive anatomical audit", "Public alternatives may be gold-calibrated",
                               "Model comparison changes encoder and cached preprocessing relative to original ResNet18"]}
    write_json(output / "preparation_status.json", summary)
    print("CPU_PREPARATION_COMPLETE", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    prepare("/kaggle/working/knee_cnn_preparation")
