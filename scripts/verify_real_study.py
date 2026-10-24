"""Replay a private Kaggle export through frozen preprocessing and the local API.

Usage: .venv/Scripts/python.exe scripts/verify_real_study.py data/local_replay
No downloads, training or GPU. Raw studies and detailed results stay under data/.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.constants import CHECKPOINT_SHA256, PIXEL_KEY, TARGETS
from app.inputs import extract_study, native_path
from app.settings import Settings
from app.worker import load_engine, sha256


def peak_memory_mb():
    """Peak working set for THIS verifier process, not aggregate server memory."""
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / (1024**2 if sys.platform == "darwin" else 1024), 1)
    import ctypes
    from ctypes import wintypes
    size = ctypes.c_size_t

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, size) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not query(current(), ctypes.byref(counters), ctypes.sizeof(counters)):
        raise ctypes.WinError()
    return round(counters.PeakWorkingSetSize / 1024**2, 1)


def score_comparison(actual, expected, tolerance):
    if set(actual) != set(TARGETS) or set(expected) != set(TARGETS):
        raise ValueError("All 12 named scores are required.")
    rows = []
    for target in TARGETS:
        left, right = float(actual[target]), float(expected[target])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (left, right)):
            raise ValueError("Scores must be finite probabilities.")
        rows.append({"finding": target, "kaggle_cpu": right, "windows_cpu": left,
                     "absolute_difference": abs(left - right)})
    maximum = max(row["absolute_difference"] for row in rows)
    if maximum > tolerance:
        raise AssertionError(f"Score parity failed: {maximum} > {tolerance}")
    return rows, maximum


def verify(folder, base_url):
    import httpx
    import numpy as np
    import pandas as pd
    import torch
    from app import raw_preprocess as raw

    settings = Settings(root=ROOT)
    reference = json.loads((folder / "reference.json").read_text())
    if reference["gpu_used"] is not False or reference["training_started"] is not False:
        raise ValueError("Expected CPU-only reference.")
    assert reference["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert reference["pixel_key"] == PIXEL_KEY
    assert reference["raw_preprocess_sha256"] == sha256(ROOT / "app" / "raw_preprocess.py")
    tolerance = 1e-5  # Declared before seeing Windows predictions; never increased after a failure.
    assert reference["local_score_absolute_tolerance"] == tolerance
    for name, key in (("study.zip", "archive_sha256"), ("series_metadata.csv", "metadata_sha256"),
                      ("reference_inputs.npz", "reference_inputs_sha256")):
        assert sha256(folder / name) == reference[key], f"Downloaded artifact changed: {name}"
    with zipfile.ZipFile(folder / "study.zip") as archive:
        assert set(archive.namelist()) == set(reference["file_sha256"])
        for member, expected in reference["file_sha256"].items():
            with archive.open(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    original = json.loads((ROOT / "models" / "original_cpu_parity.json").read_text())
    record = next(r for r in original["parity_studies"] if r["StudyInstanceUID"] == reference["study_uid"])
    assert record["image_hash"] == reference["image_and_mask_sha256"]
    assert record["packed_arrays_exact"] and record["pixels_exact"] and record["mask_exact"]
    with tempfile.TemporaryDirectory(prefix="real_replay_", dir=native_path(ROOT / ".runtime")) as temp:
        started = time.monotonic()
        uid, rows, count = extract_study(folder / "study.zip", folder / "series_metadata.csv", Path(temp) / "series", settings)
        assert uid == reference["study_uid"] and count == reference["source_file_count"]
        prep_start = time.monotonic()
        images, valid, details = raw.load_study(uid, raw.group_series(pd.DataFrame(rows)), native_path(Path(temp) / "series"))
        prep_seconds = time.monotonic() - prep_start
        with np.load(folder / "reference_inputs.npz", allow_pickle=False) as arrays:
            assert np.array_equal(images, arrays["images"]), "Windows preprocessed pixels differ."
            assert np.array_equal(valid, arrays["valid"]), "Windows window mask differs."
        assert hashlib.sha256(images.tobytes() + valid.tobytes()).hexdigest() == record["image_hash"]
        assert np.flatnonzero(valid).tolist() == reference["valid_windows"]
        assert not any(d.get("errors") or d.get("fallback_errors") for d in details)
        print(f"REAL_PIXELS_AND_MASK_EXACT: {count} DICOMs, {int(valid.sum())}/40 windows", flush=True)
        model = load_engine(settings)
        begin_inference = time.monotonic()
        with torch.inference_mode():
            scores = model(torch.from_numpy(images[None].copy()), torch.from_numpy(valid[None].copy()),
                           torch.arange(5).repeat_interleave(8)[None], chunk=8).float().sigmoid().numpy()[0]
        direct_scores = dict(zip(TARGETS, map(float, scores)))
        direct_rows, direct_difference = score_comparison(direct_scores, reference["scores"], tolerance)
        direct_inference_seconds = time.monotonic() - begin_inference
        verifier_seconds = time.monotonic() - started
        del model

    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        client.get("/health/ready").raise_for_status()
        with (folder / "study.zip").open("rb") as archive, (folder / "series_metadata.csv").open("rb") as metadata:
            response = client.post("/api/v1/jobs", data={"kind": "upload"}, files={
                "study": ("study.zip", archive, "application/zip"),
                "metadata": ("metadata.csv", metadata, "text/csv")})
        response.raise_for_status()
        job = response.json()
        route = "/api/v1/jobs/" + job["id"]
        headers = {"Authorization": "Bearer " + job["access_token"]}
        states = []
        try:
            deadline = time.monotonic() + 310
            while time.monotonic() < deadline:
                response = client.get(route, headers=headers)
                response.raise_for_status()
                state = response.json()
                if not states or states[-1] != state["state"]:
                    states.append(state["state"])
                if state["state"] in {"failed", "cancelled", "timed_out"}:
                    raise RuntimeError(state.get("error", state["state"]))
                if state["state"] == "completed":
                    break
                time.sleep(.3)
            else:
                raise RuntimeError("Local real-study request timed out.")
            response = client.get(route + "/result", headers=headers)
            response.raise_for_status()
            result = response.json()
            assert result["synthetic"] is False
            assert result["checkpoint_sha256"] == CHECKPOINT_SHA256
            assert result["valid_windows"] == reference["valid_windows"]
            comparison, api_difference = score_comparison(result["scores"], reference["scores"], tolerance)
            _, direct_api_difference = score_comparison(result["scores"], direct_scores, tolerance)
            for index in result["valid_windows"]:
                image = client.get(route + f"/previews/{index}", headers=headers)
                image.raise_for_status()
                assert image.headers["content-type"].startswith("image/png")
            assert client.get(route).status_code == 404
            (folder / "local_api_result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        finally:
            client.delete(route, headers=headers).raise_for_status()
        assert client.get(route, headers=headers).status_code == 404

    with (folder / "score_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    report = {
        "status": "REAL_STUDY_WINDOWS_REPLAY_PASS", "studies": 1, "source_files": count,
        "valid_windows": int(valid.sum()), "total_windows": 40, "target_count": len(TARGETS),
        "pixels_exact": True, "mask_exact": True, "source_file_hashes_exact": True,
        "original_kaggle_pixel_hash_exact": True, "checkpoint_sha256": CHECKPOINT_SHA256,
        "absolute_score_tolerance": tolerance, "direct_max_score_difference": direct_difference,
        "api_max_score_difference": api_difference, "direct_vs_api_max_score_difference": direct_api_difference,
        "api_states": states, "api_seconds": result["seconds"],
        "direct_preprocessing_seconds": round(prep_seconds, 3),
        "direct_inference_seconds": round(direct_inference_seconds, 3),
        "verifier_seconds_including_model_load": round(verifier_seconds, 3),
        "verifier_peak_working_set_MB": peak_memory_mb(),
        "memory_scope": "Verifier process only, not aggregate UI/server/worker memory",
        "environment": result["environment"], "api_job_and_temporary_files_cleared": True,
        "gpu_used": False, "paid_services_used": False,
        "limitation": "One study verifies implementation parity, not clinical accuracy, latency distribution or generalization."
    }
    (folder / "windows_replay_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    from urllib.parse import urlsplit
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} or endpoint.username or endpoint.password:
        parser.error("Only a loopback HTTP endpoint is allowed; this tool never uploads scans to third parties.")
    verify(args.folder.resolve(), args.base_url)
