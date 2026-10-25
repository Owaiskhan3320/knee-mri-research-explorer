"""Verify one private real study and measure repeated LOCAL CPU container requests.

Uses an existing image, read-only private inputs/weights and no container network.
No build, push, download, training, GPU or remote service. Does not alter the app.
"""
import argparse
import json
from pathlib import Path
import statistics
import uuid

from verify_container import ROOT, run


DIRECT_PROBE = r'''
import hashlib, json, math, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, '/opt/knee')
import numpy as np
import pandas as pd
import torch
from app.constants import CHECKPOINT_SHA256, PIXEL_KEY, TARGETS
from app.inputs import extract_study
from app.settings import Settings
from app.worker import load_engine, sha256
from app import raw_preprocess as raw

folder=Path('/private_replay')
settings=Settings(root=Path('/opt/knee'))
reference=json.loads((folder/'reference.json').read_text())
assert reference['gpu_used'] is False and reference['training_started'] is False
assert reference['checkpoint_sha256']==CHECKPOINT_SHA256
assert reference['pixel_key']==PIXEL_KEY
assert reference['raw_preprocess_sha256']==sha256(settings.root/'app'/'raw_preprocess.py')
tolerance=1e-5
assert reference['local_score_absolute_tolerance']==tolerance
for name,key in [('study.zip','archive_sha256'),('series_metadata.csv','metadata_sha256'),('reference_inputs.npz','reference_inputs_sha256')]:
    assert sha256(folder/name)==reference[key], 'Private input artifact changed'
with zipfile.ZipFile(folder/'study.zip') as archive:
    assert set(archive.namelist())==set(reference['file_sha256'])
    for name,expected in reference['file_sha256'].items():
        with archive.open(name) as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected, 'Source DICOM changed'
original=json.loads((settings.root/'models'/'original_cpu_parity.json').read_text())
record=next(r for r in original['parity_studies'] if r['StudyInstanceUID']==reference['study_uid'])
assert record['image_hash']==reference['image_and_mask_sha256']
assert record['packed_arrays_exact'] and record['pixels_exact'] and record['mask_exact']
with tempfile.TemporaryDirectory(dir=settings.root/'.runtime') as temporary:
    uid,rows,count=extract_study(folder/'study.zip',folder/'series_metadata.csv',Path(temporary)/'series',settings)
    assert uid==reference['study_uid'] and count==reference['source_file_count']
    images,valid,details=raw.load_study(uid,raw.group_series(pd.DataFrame(rows)),Path(temporary)/'series')
    with np.load(folder/'reference_inputs.npz',allow_pickle=False) as arrays:
        assert np.array_equal(images,arrays['images']), 'Linux processed pixels differ'
        assert np.array_equal(valid,arrays['valid']), 'Linux validity mask differs'
    assert hashlib.sha256(images.tobytes()+valid.tobytes()).hexdigest()==record['image_hash']
    assert np.flatnonzero(valid).tolist()==reference['valid_windows']
    assert not any(d.get('errors') or d.get('fallback_errors') for d in details)
    model=load_engine(settings)
    with torch.inference_mode():
        values=model(torch.from_numpy(images[None].copy()),torch.from_numpy(valid[None].copy()),
                     torch.arange(5).repeat_interleave(8)[None],chunk=8).float().sigmoid().numpy()[0]
    expected=reference['scores']
    assert set(expected)==set(TARGETS) and len(values)==12
    assert all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in values)
    maximum=max(abs(float(v)-float(expected[t])) for t,v in zip(TARGETS,values))
    assert maximum<=tolerance, 'Linux direct score parity failed'
print('BENCHMARK_JSON:'+json.dumps({'source_files':count,'valid_windows':int(valid.sum()),
    'target_count':12,'source_file_hashes_exact':True,'pixels_exact':True,'mask_exact':True,
    'original_cached_pixel_hash_exact':True,'absolute_score_tolerance':tolerance,
    'direct_max_score_difference':maximum,'cpu_threads':torch.get_num_threads()}))
'''


API_PROBE = r'''
import importlib.metadata, json, math, os, sys, time
from pathlib import Path
import httpx
folder=Path('/private_replay')
reference=json.loads((folder/'reference.json').read_text())
repeats=int(sys.argv[1])
tolerance=1e-5
assert reference['local_score_absolute_tolerance']==tolerance
assert os.getuid()==1000
cgroup=Path('/sys/fs/cgroup')
assert (cgroup/'memory.peak').is_file(), 'cgroup v2 peak-memory measurement unavailable'
assert int((cgroup/'memory.max').read_text())==2*1024**3
assert (cgroup/'cpu.max').read_text().split()==['200000','100000']

def events():
    return {k:int(v) for k,v in (line.split() for line in (cgroup/'memory.events').read_text().splitlines())}

with httpx.Client(base_url='http://127.0.0.1:7860',timeout=30,trust_env=False) as client:
    wait_started=time.monotonic()
    while time.monotonic()-wait_started<90:
        try:
            ready=client.get('/health/ready')
            if ready.status_code==200:
                break
            if ready.json().get('error'):
                raise RuntimeError('Model startup failed')
        except httpx.TransportError:
            pass
        time.sleep(.2)
    else:
        raise RuntimeError('Container did not become ready')
    readiness_wait=round(time.monotonic()-wait_started,3)
    baseline=int((cgroup/'memory.current').read_text())
    requests=[]
    first_scores=None
    for index in range(1+repeats):
        started=time.monotonic()
        with (folder/'study.zip').open('rb') as archive,(folder/'series_metadata.csv').open('rb') as metadata:
            response=client.post('/api/v1/jobs',data={'kind':'upload'},files={
                'study':('study.zip',archive,'application/zip'),'metadata':('series_metadata.csv',metadata,'text/csv')})
        response.raise_for_status()
        job=response.json()
        route='/api/v1/jobs/'+job['id']
        headers={'Authorization':'Bearer '+job['access_token']}
        try:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                response=client.get(route,headers=headers)
                response.raise_for_status()
                state=response.json()
                if state['state'] in {'failed','cancelled'}:
                    raise RuntimeError('Real-study API processing failed')
                if state['state']=='completed':
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('Real-study API timed out')
            response=client.get(route+'/result',headers=headers)
            response.raise_for_status()
            result=response.json()
            elapsed=time.monotonic()-started
            assert result['synthetic'] is False
            assert result['checkpoint_sha256']==reference['checkpoint_sha256']
            assert result['preprocessing_pixel_key']==reference['pixel_key']
            assert result['source_files']==reference['source_file_count']
            assert result['valid_windows']==reference['valid_windows']
            actual,expected=result['scores'],reference['scores']
            assert set(actual)==set(expected) and len(actual)==12
            assert all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in actual.values())
            maximum=max(abs(actual[t]-expected[t]) for t in expected)
            assert maximum<=tolerance, 'Linux API score parity failed'
            if first_scores is None:
                first_scores=actual
            repeated_difference=max(abs(actual[t]-first_scores[t]) for t in expected)
            assert repeated_difference<=tolerance
            if index==0:
                assert client.get(route).status_code==404
                for window in result['valid_windows']:
                    image=client.get(route+f'/previews/{window}',headers=headers)
                    image.raise_for_status()
                    assert image.headers['content-type'].startswith('image/png')
                assert client.get(route+'/previews/'+str(result['valid_windows'][0])).status_code==404
            requests.append({'request':index+1,'phase':'first_request' if index==0 else 'subsequent_request',
                'worker_seconds':result['seconds'],'upload_to_result_seconds':round(elapsed,3),
                'max_score_difference_from_kaggle':maximum,'max_score_difference_from_first':repeated_difference})
        finally:
            client.delete(route,headers=headers).raise_for_status()
        assert client.get(route,headers=headers).status_code==404
    assert not list(Path('/opt/knee/.runtime/jobs').iterdir()), 'Job files remain after deletion'
    memory_events=events()
    assert memory_events.get('oom',0)==0 and memory_events.get('oom_kill',0)==0
    peak=int((cgroup/'memory.peak').read_text())
    assert peak<2*1024**3
    report={'requests':requests,'readiness_wait_seconds_from_probe_start':readiness_wait,
        'memory_current_before_requests_bytes':baseline,'memory_peak_bytes':peak,
        'memory_current_after_requests_bytes':int((cgroup/'memory.current').read_text()),
        'memory_events':memory_events,'memory_swap_peak_bytes':int((cgroup/'memory.swap.peak').read_text()) if (cgroup/'memory.swap.peak').exists() else None,
        'all_32_previews_and_access_checks_passed':True,'all_api_jobs_cleared':True,
        'environment':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','pydicom']}}
print('BENCHMARK_JSON:'+json.dumps(report))
'''


def probe_result(output):
    records = [line.removeprefix("BENCHMARK_JSON:") for line in output.splitlines()
               if line.startswith("BENCHMARK_JSON:")]
    if len(records) != 1:
        raise RuntimeError("Expected exactly one successful verification report.")
    return json.loads(records[0])


def benchmark(folder, context, image, warm_repeats, output):
    docker = ["docker", "--context", context]
    endpoint = json.loads(run(["docker", "context", "inspect", context, "--format", "{{json .Endpoints.docker.Host}}"], timeout=15))
    if not endpoint.startswith(("npipe://", "unix://")):
        raise ValueError("Only a local Docker engine is allowed.")
    info = json.loads(run(docker + ["image", "inspect", image]))[0]
    if info["Os"] != "linux" or info["Config"]["User"] != "researcher":
        raise ValueError("Expected the Linux non-root application image.")
    folder = folder.resolve()
    for path in [folder / name for name in ("study.zip", "series_metadata.csv", "reference.json", "reference_inputs.npz")] + [ROOT / "models" / name for name in ("deploy.pt", "manifest.json", "original_cpu_parity.json")]:
        if not path.is_file():
            raise ValueError("The authorized local replay/model bundle is incomplete.")
    options = ["--init", "--network", "none", "--read-only", "--cpus", "2", "--memory", "2g",
               "--memory-swap", "2g", "--pids-limit", "256", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,noexec,size=128m",
               "--tmpfs", "/opt/knee/.runtime:rw,nosuid,noexec,size=512m,uid=1000,gid=1000",
               "--mount", f"type=bind,source={ROOT / 'models'},target=/opt/knee/models,readonly",
               "--mount", f"type=bind,source={folder},target=/private_replay,readonly"]
    prefix = "knee-replay-" + uuid.uuid4().hex[:10]
    observations = {}
    for phase, probe in (("direct_parity", DIRECT_PROBE), ("api_measurement", API_PROBE)):
        identifier = None
        try:
            entry = ["--entrypoint", "python"] if phase == "direct_parity" else ["-e", "KNEE_ALLOW_UPLOADS=1"]
            command = ["-c", "import time; time.sleep(600)"] if phase == "direct_parity" else []
            identifier = run(docker + ["run", "-d", "--name", prefix + "-" + phase] + options + entry + [image] + command)
            script_args = [] if phase == "direct_parity" else [str(warm_repeats)]
            result = run(docker + ["exec", "-i", identifier, "python", "-"] + script_args,
                         input=probe, timeout=150 if phase == "direct_parity" else 120 * (warm_repeats + 1))
            observations[phase] = probe_result(result)
            state = json.loads(run(docker + ["inspect", identifier]))[0]
            assert state["State"]["Running"] and not state["State"]["OOMKilled"]
            assert state["HostConfig"]["NetworkMode"] == "none"
            assert state["HostConfig"]["ReadonlyRootfs"] is True
            assert state["HostConfig"]["Memory"] == state["HostConfig"]["MemorySwap"] == 2 * 1024**3
            assert state["HostConfig"]["NanoCpus"] == 2_000_000_000
            assert all(not m["RW"] for m in state["Mounts"] if m["Type"] == "bind")
            print("LOCAL_REAL_STUDY_PHASE_PASS:", phase, flush=True)
        finally:
            if identifier:
                run(docker + ["rm", "-f", identifier])
    measurement = observations["api_measurement"]
    subsequent = measurement["requests"][1:]
    totals = [r["worker_seconds"]["total"] for r in subsequent]
    report = {"status": "LOCAL_LINUX_REAL_STUDY_REPLAY_PASS", "unique_studies": 1,
              "image_id": info["Id"], "platform": info["Os"] + "/" + info["Architecture"],
              "cpu_limit": 2, "memory_limit_bytes": 2 * 1024**3, "container_swap_disabled": True,
              "network_mode": "none", **observations,
              "subsequent_worker_total_seconds": {"count": len(totals), "median": round(statistics.median(totals), 3),
                                                   "minimum": min(totals), "maximum": max(totals)},
              "memory_scope": "cgroup v2 lifetime peak of API container: server, worker, HTTP probe, filesystem cache and tmpfs; excludes separate direct parity container, Docker VM, host and browser",
              "timing_scope": "Worker processing excludes upload, queue wait and model startup; upload-to-result includes local multipart transfer and polling; readiness wait begins when the probe starts",
              "gpu_used": False, "paid_services_used": False, "temporary_containers_removed": True,
              "limitation": "One study repeated on a shared Docker Desktop VM. No clinical/accuracy evaluation, diverse-study coverage, tail-latency estimate or minimum hosting specification."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Authorized private local replay folder")
    parser.add_argument("--context", default="desktop-linux")
    parser.add_argument("--image", default="knee-mri-explorer:local")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "linux_real_study_report.json")
    args = parser.parse_args()
    if not 1 <= args.warm_repeats <= 10:
        parser.error("Use between 1 and 10 subsequent requests.")
    benchmark(args.folder, args.context, args.image, args.warm_repeats, args.output)
