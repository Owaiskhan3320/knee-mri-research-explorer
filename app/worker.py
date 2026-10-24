"""Persistent CPU inference worker. Never trains or contacts a remote service."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .constants import CHECKPOINT_SHA256, MODEL_ID, NOTICE, PIXEL_KEY, SLOT_NAMES, TARGETS


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_engine(settings):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from .model import StudyMIL
    if not settings.model_path.is_file():
        raise RuntimeError("Model checkpoint is missing. See models/README.md.")
    if sha256(settings.model_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint identity check failed; use the verified epoch-8 model.")
    manifest = json.loads((settings.root / "models" / "manifest.json").read_text())
    for filename, key in (("model.py", "model_module_sha256"), ("raw_preprocess.py", "raw_preprocess_sha256")):
        if sha256(settings.root / "app" / filename) != manifest[key]:
            raise RuntimeError("Model architecture or verified preprocessing source has changed.")
    torch.set_num_threads(settings.cpu_threads)
    ck = torch.load(settings.model_path, map_location="cpu", weights_only=True)
    if ck.get("selected_epoch") != 8 or ck.get("targets") != TARGETS or ck.get("trained") is not True:
        raise RuntimeError("Checkpoint metadata does not match the deployed model.")
    model = StudyMIL(num_targets=12, slots=5, projection=256, pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().requires_grad_(False)
    return model


def process_job(settings, model, task, emit):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from . import raw_preprocess as raw
    from .fixtures import create_fixture
    from .inputs import extract_study, native_path
    job_id, directory, kind = task["id"], native_path(task["directory"]), task["kind"]
    started = time.monotonic()
    emit({"id": job_id, "state": "validating", "progress": .1})
    if kind == "synthetic":
        archive, metadata = create_fixture(directory)
    else:
        archive, metadata = directory / "study.zip", directory / "series_metadata.csv"
    uid, rows, files = extract_study(archive, metadata, directory / "series", settings)
    emit({"id": job_id, "state": "preprocessing", "progress": .3})
    grouped = raw.group_series(pd.DataFrame(rows))
    prep_start = time.monotonic()
    images, valid, details = raw.load_study(uid, grouped, directory / "series")
    prep_seconds = time.monotonic() - prep_start
    previews = directory / "previews"
    previews.mkdir(exist_ok=True)
    for i in np.flatnonzero(valid):
        Image.fromarray(images[i, 1]).save(previews / f"{int(i)}.png")
    emit({"id": job_id, "state": "inferencing", "progress": .7})
    x = torch.from_numpy(images[None].copy())
    mask = torch.from_numpy(valid[None].copy())
    slots = torch.arange(5).repeat_interleave(8)[None]
    infer_start = time.monotonic()
    with torch.inference_mode():
        logits = model(x, mask, slots, chunk=8)
        scores = logits.float().sigmoid().numpy()[0]
    if logits.shape != (1, 12) or not np.isfinite(scores).all():
        raise RuntimeError("The model produced an invalid result; no default prediction was substituted.")
    warnings = []
    for detail in details:
        if detail.get("missing"):
            warnings.append(f"Missing input slot: {SLOT_NAMES[detail['slot']]}")
        elif detail.get("errors") or detail.get("fallback_errors"):
            warnings.append(f"Input decoding/order warnings in slot {detail['slot'] + 1}; inspect the source study.")
    if kind == "synthetic":
        warnings.insert(0, "Synthetic engineering fixture: these scores have no medical meaning and do not measure accuracy.")
    result = dict(
        job_id=job_id, model_id=MODEL_ID, checkpoint_sha256=CHECKPOINT_SHA256,
        model_public_macro_auc=.833, score_type="uncalibrated_single_model_sigmoid",
        notice=NOTICE, synthetic=kind == "synthetic", scores=dict(zip(TARGETS, map(float, scores))),
        valid_windows=np.flatnonzero(valid).tolist(), total_windows=40, source_files=files,
        preprocessing_pixel_key=PIXEL_KEY,
        seconds=dict(preprocessing=round(prep_seconds, 3), inference=round(time.monotonic() - infer_start, 3),
                     total=round(time.monotonic() - started, 3)),
        warnings=warnings,
        environment={n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "Pillow", "pydicom")},
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def worker_main(settings, tasks, messages):
    try:
        model = load_engine(settings)
        messages.put({"event": "ready"})
    except Exception as exc:
        messages.put({"event": "startup_error", "error": str(exc)})
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        try:
            result = process_job(settings, model, task, messages.put)
            messages.put({"id": task["id"], "state": "completed", "progress": 1., "result": result})
        except ValueError as exc:
            messages.put({"id": task["id"], "state": "failed", "error": str(exc)})
        except Exception:
            # Do not send raw headers, identifiers, filenames or stack traces to clients/logs.
            messages.put({"id": task["id"], "state": "failed", "error": "Study processing failed. Check DICOM support, spacing and series metadata. No prediction was generated."})
