"""Test an existing image on a LOCAL Docker engine. Never builds/pushes images.

Private weights are mounted read-only; no MRI data is used. Temporary test
containers are removed in finally blocks. The local image remains available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, input=None, timeout=60):
    completed = subprocess.run(command, input=input, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
    return completed.stdout.strip()


MISSING_PROBE = r'''
import json, os, time, urllib.request, urllib.error
assert os.getuid() == 1000
def get(path):
    try:
        with urllib.request.urlopen('http://127.0.0.1:7860'+path,timeout=3) as response:
            return response.status,json.load(response)
    except urllib.error.HTTPError as error:
        return error.code,json.load(error)
deadline=time.monotonic()+90
while time.monotonic()<deadline:
    try:
        status,value=get('/health/ready')
        if status==503 and 'missing' in str(value.get('error','')).lower():
            break
    except OSError:
        pass
    time.sleep(.5)
else:
    raise RuntimeError('Expected a missing-model readiness failure')
assert get('/health/live')[0]==200
assert value['ready'] is False
request=urllib.request.Request('http://127.0.0.1:7860/api/v1/jobs',data=b'kind=synthetic',headers={'Content-Type':'application/x-www-form-urlencoded'})
try:
    urllib.request.urlopen(request,timeout=3)
except urllib.error.HTTPError as error:
    assert error.code==429
    rejected_status=error.code
else:
    raise AssertionError('A missing model must reject inference requests')
print(json.dumps({'missing_weights_readiness_status':status,'inference_request_rejected_status':rejected_status,'uid':os.getuid()}))
'''


IMAGE_PROBE = r'''
import hashlib,json
from pathlib import Path
root=Path('/opt/knee')
assert {p.name for p in root.iterdir()}=={'app','models','.runtime','run.py','requirements.txt','requirements-lock.txt'}
assert not list((root/'models').iterdir())
assert not list((root/'.runtime').iterdir())
assert all(p.is_file() and p.suffix=='.py' for p in (root/'app').iterdir())
print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'app').glob('*.py'))}))
'''


def verify(context, image, port, output):
    docker = ["docker", "--context", context]
    endpoint = json.loads(run(["docker", "context", "inspect", context, "--format", "{{json .Endpoints.docker.Host}}"], timeout=15))
    if not endpoint.startswith(("npipe://", "unix://")):
        raise ValueError("Only a local Docker engine is allowed; no remote or cloud context.")
    info = json.loads(run(docker + ["image", "inspect", image]))[0]
    if info["Os"] != "linux" or info["Config"]["User"] != "researcher":
        raise ValueError("Expected the Linux non-root research image.")
    if not (ROOT / "models" / "deploy.pt").is_file():
        raise ValueError("Import the authorized model before the positive container check.")
    image_sources = json.loads(run(docker + ["run", "--rm", "--network", "none", "--read-only",
                                           "-i", "--entrypoint", "python", image, "-"], input=IMAGE_PROBE))
    host_sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT / "app").glob("*.py"))}
    if image_sources != host_sources:
        raise ValueError("Image application files differ from this source checkout.")
    smoke = (ROOT / "scripts" / "smoke_api.py").read_text()
    prefix = "knee-verify-" + uuid.uuid4().hex[:10]
    common = ["--init", "--read-only", "--cpus", "2", "--memory", "2g", "--pids-limit", "256",
              "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
              "--health-start-period", "0s", "--health-interval", "2s", "--health-retries", "1",
              "--tmpfs", "/tmp:rw,nosuid,noexec,size=128m",
              "--tmpfs", "/opt/knee/.runtime:rw,nosuid,noexec,size=512m,uid=1000,gid=1000"]
    mount = "type=bind,source=" + str((ROOT / "models").resolve()) + ",target=/opt/knee/models,readonly"
    observations = {}
    started = time.monotonic()
    for phase in ("missing", "offline", "localhost"):
        name = prefix + "-" + phase
        network = ["--network", "none"] if phase != "localhost" else ["-p", f"127.0.0.1:{port}:7860"]
        options = common + network + ([] if phase == "missing" else ["--mount", mount])
        identifier = None
        try:
            identifier = run(docker + ["run", "-d", "--name", name] + options + [image])
            if phase == "missing":
                result = run(docker + ["exec", "-i", identifier, "python", "-"], input=MISSING_PROBE, timeout=110)
                observations[phase] = json.loads(result)
            else:
                report = run(docker + ["exec", "-i", identifier, "python", "-", "http://127.0.0.1:7860", "--uploads-disabled"], input=smoke, timeout=130)
                if "LOCAL_END_TO_END_PASS" not in report or "LOCAL_UPLOAD_DISABLED_POLICY_PASS" not in report:
                    raise RuntimeError("Actual model/API smoke check did not pass.")
                print(report, flush=True)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    state = json.loads(run(docker + ["inspect", identifier]))[0]
                    if state["State"]["Health"]["Status"] == "healthy":
                        break
                    time.sleep(.5)
                else:
                    raise RuntimeError("Container readiness healthcheck did not become healthy.")
                assert state["HostConfig"]["ReadonlyRootfs"] is True
                assert state["HostConfig"]["Memory"] == 2 * 1024**3
                assert state["HostConfig"]["NanoCpus"] == 2_000_000_000
                assert state["State"]["OOMKilled"] is False
                bound = next(item for item in state["Mounts"] if item["Destination"] == "/opt/knee/models")
                assert bound["RW"] is False
                if phase == "localhost":
                    from urllib.request import urlopen
                    with urlopen(f"http://127.0.0.1:{port}/health/ready", timeout=5) as response:
                        assert response.status == 200 and json.load(response)["ready"] is True
                    published = state["NetworkSettings"]["Ports"]["7860/tcp"]
                    assert len(published) == 1 and published[0]["HostIp"] == "127.0.0.1"
                statistics = json.loads(run(docker + ["stats", "--no-stream", "--format", "{{json .}}", identifier]))
                observations[phase] = {"healthy": True, "read_only_root": True, "read_only_weights": True,
                    "model_inference_and_access_checks_passed": True, "upload_policy_403": True,
                    "oom_killed": False, "memory_snapshot": statistics["MemUsage"],
                    "network_mode": state["HostConfig"]["NetworkMode"], "cpu_limit": 2, "memory_limit_GiB": 2}
            print("CONTAINER_PHASE_PASS:", phase, flush=True)
        finally:
            if identifier:
                run(docker + ["rm", "-f", identifier])
    report = {"status": "LOCAL_DOCKER_VERIFICATION_PASS", "image_id": info["Id"],
              "image_size_bytes": info["Size"], "platform": info["Os"] + "/" + info["Architecture"],
              "image_content": {"application_source_files_verified": len(image_sources), "private_weights_baked_in": False,
                                "runtime_files_baked_in": False},
              "elapsed_seconds": round(time.monotonic()-started,3), "phases": observations,
              "gpu_used": False, "paid_services_used": False, "temporary_containers_removed": True,
              "limitation": "Generated DICOM engineering checks only; memory values are post-request snapshots, not peak benchmarks."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="desktop-linux")
    parser.add_argument("--image", default="knee-mri-explorer:local")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "container_verification.json")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged local port between 1024 and 65535.")
    verify(args.context, args.image, args.port, args.output)
