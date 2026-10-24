"""Exercise the actual local app and isolated model worker with synthetic DICOMs."""
import json
import sys
import time
from pathlib import Path
import httpx
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.fixtures import create_fixture

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
deadline = time.monotonic() + 100
with httpx.Client(base_url=base, timeout=20, trust_env=False) as client:
    while time.monotonic() < deadline:
        try:
            if client.get("/health/ready").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(.5)
    else:
        raise RuntimeError("Local model did not become ready")
    response = client.post("/api/v1/jobs", data={"kind": "synthetic"})
    response.raise_for_status()
    job = response.json()
    headers = {"Authorization": "Bearer " + job["access_token"]}
    route = "/api/v1/jobs/" + job["id"]
    assert client.get(route).status_code == 404
    states = []
    while time.monotonic() < deadline:
        response = client.get(route, headers=headers)
        response.raise_for_status()
        state = response.json()
        if not states or states[-1] != state["state"]:
            states.append(state["state"])
        if state["state"] in {"failed", "cancelled"}:
            raise RuntimeError(state)
        if state["state"] == "completed":
            break
        time.sleep(.3)
    else:
        raise RuntimeError("Synthetic job did not complete")
    response = client.get(route + "/result", headers=headers)
    response.raise_for_status()
    result = response.json()
    assert result["synthetic"] is True and len(result["scores"]) == 12
    assert len(result["valid_windows"]) == 40
    assert client.get(route + "/previews/0", headers=headers).status_code == 200
    assert client.get(route + "/previews/0").status_code == 404
    print(json.dumps({"status": "LOCAL_END_TO_END_PASS", "states": states, "model_id": result["model_id"],
                      "seconds": result["seconds"], "windows": len(result["valid_windows"]),
                      "checkpoint_sha256": result["checkpoint_sha256"], "synthetic_not_accuracy_evaluation": True}, indent=2))
    client.delete(route, headers=headers).raise_for_status()
    assert client.get(route, headers=headers).status_code == 404
    with tempfile.TemporaryDirectory(prefix="knee_local_smoke_") as temporary:
        archive, metadata = create_fixture(Path(temporary))
        with archive.open("rb") as z, metadata.open("rb") as f:
            upload = client.post("/api/v1/jobs", data={"kind": "upload"},
                                 files={"study": ("study.zip", z, "application/zip"),
                                        "metadata": ("metadata.csv", f, "text/csv")})
        upload.raise_for_status()
        other = upload.json()
        route = "/api/v1/jobs/" + other["id"]
        headers = {"Authorization": "Bearer " + other["access_token"]}
        while time.monotonic() < deadline:
            state = client.get(route, headers=headers).json()
            if state["state"] == "failed":
                raise RuntimeError(state)
            if state["state"] == "completed":
                break
            time.sleep(.3)
        uploaded_result = client.get(route + "/result", headers=headers)
        uploaded_result.raise_for_status()
        uploaded_result = uploaded_result.json()
        assert uploaded_result["scores"] == result["scores"], "Upload and built-in inputs must use identical inference"
        assert uploaded_result["synthetic"] is False
        client.delete(route, headers=headers).raise_for_status()
        print("LOCAL_MULTIPART_UPLOAD_PARITY_PASS (generated data only)")
