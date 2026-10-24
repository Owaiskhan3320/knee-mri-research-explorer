"""Raw DICOM equivalent of the frozen five-slot, 40-window pixel cache.

The numerical body of ``read_windows`` is preserved from cache cell 3.
Only study metadata and the train/test series directory are injected. This
module returns uint8 pixels; the saved ConvNeXt model performs /255 and
ImageNet normalization itself. Do not normalize these arrays twice.

This module never installs packages, downloads assets, trains, or uses CUDA.
"""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import pydicom


CFG = dict(size=224, crop_mm=140.0, windows_per_slot=8)
SLOTS = [("Sagittal", True), ("Coronal", True), ("Axial", True),
         ("Sagittal", False), ("Coronal", False)]
SLOT_IDS = np.repeat(np.arange(5, dtype=np.int64), 8)
EXPECTED_PIXEL_KEY = "2d0240242985abb81c302fdc9f134933769e3d1032b62f35eed3be9b6f390c79"
DECODER_VERSIONS = {
    "pydicom": "3.0.2", "pylibjpeg": "2.1.0",
    "pylibjpeg-libjpeg": "2.4.0", "pylibjpeg-openjpeg": "2.6.0",
}
SOURCE_PROVENANCE = {
    "cache_notebook_sha256": "bf772983d1f1072fa5f12272f77b3027f76b0f2ff06db92b67680ac1aa9c21d8",
    "cache_cell_02_source_sha256": "e3a1424f41697f5155ec0f3c2d9c4357525edd4d5d930d080539e41d21c96c8c",
    "cache_cell_03_source_sha256": "4a819c7236a8ea36cb84b3da42acc8e9ff4b8b97afeb0a519a1a1d43214803fb",
    "cache_execution_utc": "2026-10-15T23:03:17.963563+00:00/2026-10-15T23:54:41.165091+00:00",
    "original_preparation_version": 358823808,
}


def as_bool(value):
    text = str(value).strip().lower()
    assert text in {"true", "false", "1", "0"}, f"Unexpected sequence flag: {text}"
    return text in {"true", "1"}


def normalize_series(series):
    """Copy a train_series.csv/test_series.csv frame and derive original flags.

    Read CSV UID columns as strings before calling this function. Missing or
    duplicate SeriesInstanceUID metadata is a hard error, never guessed.
    """
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Fat_Suppression",
                "Fluid_Sensitive", "Anatomical_Plane"}
    assert required <= set(series.columns), "Incomplete series metadata"
    series = series.copy()
    for name in ("StudyInstanceUID", "SeriesInstanceUID"):
        assert series[name].notna().all() and series[name].map(lambda x: isinstance(x, str)).all()
        assert series[name].str.len().gt(0).all(), f"Empty {name}"
    assert series.SeriesInstanceUID.is_unique, "Duplicate SeriesInstanceUID metadata"
    series["fs"] = series.Fat_Suppression.map(as_bool)
    series["fluid"] = series.Fluid_Sensitive.map(as_bool)
    series["plane"] = series.Anatomical_Plane.astype(str).str.strip().str.title()
    return series


def group_series(series):
    normalized = normalize_series(series)
    return {uid: frame for uid, frame in normalized.groupby("StudyInstanceUID", sort=False)}


def read_windows(folder):
    records, errors = [], []
    for path in sorted(folder.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if int(ds.get("NumberOfFrames", 1)) != 1:
                raise ValueError("Multiframe input needs a separate verified loader")
            records.append(dict(path=path, iop=np.asarray(ds.get("ImageOrientationPatient", []), float),
                ipp=np.asarray(ds.get("ImagePositionPatient", []), float),
                instance=float(ds.get("InstanceNumber", len(records))),
                spacing=np.asarray(ds.get("PixelSpacing", []), float)))
        except Exception as exc:
            errors.append(f"header:{path.name}:{type(exc).__name__}")
    if len(records) < 3:
        raise ValueError("Fewer than three readable headers")
    reference = records[0]["iop"]
    geometry = len(reference) == 6 and all(len(r["iop"]) == 6 and len(r["ipp"]) == 3
        and np.allclose(r["iop"], reference, atol=1e-3) for r in records)
    normal = np.cross(reference[:3], reference[3:]) if geometry else None
    geometry = geometry and np.linalg.norm(normal) > 0.5
    if geometry:
        records.sort(key=lambda r: (float(np.dot(r["ipp"], normal)), str(r["path"])))
    else:
        records.sort(key=lambda r: (r["instance"], str(r["path"])))
        errors.append("ordering:InstanceNumber_fallback")
    # Do not remove failed pixel slices: doing so would create false neighbors across a gap.
    arrays = []
    for r in records:
        try:
            if len(r["spacing"]) != 2 or not np.isfinite(r["spacing"]).all() or (r["spacing"] <= 0).any():
                raise ValueError("Missing or invalid physical PixelSpacing")
            ds = pydicom.dcmread(r["path"])
            a = ds.pixel_array.astype(np.float32)
            assert a.ndim == 2
            a = a * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
            if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
                a = -a
            assert np.isfinite(a).all()
            arrays.append(a)
        except Exception as exc:
            arrays.append(None)
            errors.append(f"pixel:{r['path'].name}:{type(exc).__name__}")
    decoded = [a for a in arrays if a is not None]
    if len(decoded) < 3:
        raise ValueError("Fewer than three valid decoded slices; inspect decoders/spacing")
    # Uniform stride sampling limits percentile memory while preserving series-level scaling.
    lo, hi = np.percentile(np.concatenate([a[::4, ::4].ravel() for a in decoded]), [1, 99])
    if hi <= lo:
        raise ValueError("Constant series")
    windows = np.zeros((CFG["windows_per_slot"], 3, CFG["size"], CFG["size"]), np.uint8)
    valid = np.zeros(CFG["windows_per_slot"], bool)
    anchors = np.rint(np.linspace(1, len(records)-2, CFG["windows_per_slot"])).astype(int)
    for w, center in enumerate(anchors):
        neighbors = [center-1, center, center+1]
        if any(arrays[j] is None for j in neighbors):
            continue
        if geometry:
            distances = np.diff([float(np.dot(records[j]["ipp"], normal)) for j in neighbors])
            if (distances <= 1e-5).any():
                errors.append("window:duplicate_position")
                continue
        for channel, j in enumerate(neighbors):
            a = arrays[j]
            row_mm, col_mm = records[j]["spacing"]
            ch, cw = max(2, round(CFG["crop_mm"]/row_mm)), max(2, round(CFG["crop_mm"]/col_mm))
            height, width = a.shape
            left, top = (width-cw)//2, (height-ch)//2
            image = Image.fromarray(np.clip((a-lo)/(hi-lo), 0, 1).astype(np.float32))
            image = image.crop((left, top, left+cw, top+ch))  # zero-pad outside FOV
            image = image.resize((CFG["size"], CFG["size"]), Image.Resampling.BILINEAR)
            windows[w, channel] = np.rint(np.clip(np.asarray(image), 0, 1)*255).astype(np.uint8)
        valid[w] = True
    if not valid.any():
        raise ValueError("No valid neighboring-slice windows")
    return windows, valid, dict(slices=len(records), decoded=len(decoded), geometric_order=bool(geometry),
        spacing_mm=records[len(records)//2]["spacing"].tolist(), errors=errors)


def load_study(uid, series_by_study, series_root):
    """Load one study, preserving all 40 positions and explicit missing masks.

    ``series_root`` must be competition/train_series or competition/test_series.
    Metadata must have been produced by group_series/normalize_series. Missing
    studies and all-invalid bags fail rather than receiving a default prediction.
    """
    uid = str(uid)
    series_root = Path(series_root)
    shape = (len(SLOTS), CFG["windows_per_slot"])
    images = np.zeros((*shape, 3, CFG["size"], CFG["size"]), np.uint8)
    valid = np.zeros(shape, bool)
    details = []
    for slot, (plane, fs) in enumerate(SLOTS):
        candidates = series_by_study[uid]
        candidates = candidates.loc[candidates.plane.eq(plane) & candidates.fs.eq(fs)]
        candidates = candidates.sort_values(["fluid", "SeriesInstanceUID"], ascending=[False, True])
        slot_errors = []
        for row in candidates.itertuples(index=False):
            folder = series_root / uid / row.SeriesInstanceUID
            try:
                x, mask, info = read_windows(folder)
                images[slot], valid[slot] = x, mask
                details.append(dict(slot=slot, plane=plane, fs=fs, series=row.SeriesInstanceUID,
                    fallback_errors=slot_errors, **info))
                break
            except Exception as exc:
                slot_errors.append(f"{row.SeriesInstanceUID}:{type(exc).__name__}:{exc}")
        if not valid[slot].any():
            details.append(dict(slot=slot, plane=plane, fs=fs, missing=True, errors=slot_errors))
    assert valid.any(), f"All slots invalid for study {uid}; no default prediction allowed"
    return images.reshape(-1, 3, CFG["size"], CFG["size"]), valid.ravel(), details


def pack_arrays(uid, images, valid, pixel_key=EXPECTED_PIXEL_KEY):
    """Reproduce cache array packing exactly, including invalid all-zero refs.

    This is intended for byte/array parity checks, not a new inference cache.
    ``frames[refs]`` restores the dense input. NPZ ZIP timestamps are deliberately
    excluded from this contract; compare frames/refs/valid/image_hash instead.
    """
    assert images.shape == (40, 3, 224, 224) and images.dtype == np.uint8
    assert valid.shape == (40,) and valid.dtype == bool and valid.any()
    assert not images[~valid].any(), "Invalid windows must remain zero"
    unique = [np.zeros((CFG["size"], CFG["size"]), np.uint8)]
    lookup = {unique[0].tobytes(): 0}
    refs = np.zeros((len(images), 3), np.int16)
    for window in np.flatnonzero(valid):
        for channel in range(3):
            value = images[window, channel].tobytes()
            index = lookup.get(value)
            if index is None:
                index = len(unique); lookup[value] = index
                unique.append(images[window, channel].copy())
            refs[window, channel] = index
    stack = np.stack(unique)
    assert np.array_equal(stack[refs], images), "Lossless packing round trip failed"
    return dict(frames=stack, refs=refs, valid=valid.copy(), uid=np.asarray(str(uid)),
                key=np.asarray(pixel_key),
                image_hash=np.asarray(hashlib.sha256(images.tobytes()+valid.tobytes()).hexdigest()))
