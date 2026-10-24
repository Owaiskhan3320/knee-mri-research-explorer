"""Import verified private model assets; never replace application source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "e30535be63503f489d3e0961afa51cce83bfe7d9766faffbe5dfc2be454310dd"
PREFIX = "knee_convnext_raw_parity/bundle/"
RAW_SHA = "69c185efb9f060d63463434c3e372e7a2c3a668dc33fbf7a77ccf13f589b01ed"
ORIGINAL_MODEL_SHA = "2b48dbe18a32e4a5e9c8c9adb83250e0f399c49343ff5cda6b0155bd855f29cd"
MODEL_SHA = "1523d4957f3c1e80e135bc5875866af5e35607397757eaaad253d9c79f32fa6e"


def read_member(archive, name, maximum):
    members = [i for i in archive.infolist() if i.filename == PREFIX + name]
    if len(members) != 1 or members[0].file_size > maximum or members[0].flag_bits & 1:
        raise ValueError(f"Missing, duplicate, encrypted or oversized bundle member: {name}")
    return archive.read(members[0])


def validate_bundle(archive_path, root=ROOT):
    for name, expected in (("raw_preprocess.py", RAW_SHA), ("model.py", MODEL_SHA)):
        if hashlib.sha256((root / "app" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Installed source identity check failed. Restore the verified source first.")
    manifest = json.loads((root / "models" / "manifest.json").read_text())
    if manifest.get("checkpoint_sha256") != EXPECTED:
        raise ValueError("Installed model manifest does not identify the expected epoch-8 model.")
    with zipfile.ZipFile(archive_path) as archive:
        raw = read_member(archive, "raw_preprocess.py", 128 * 1024)
        original = read_member(archive, "cnn_convnext.py", 512 * 1024)
        for content, expected in ((raw, RAW_SHA), (original, ORIGINAL_MODEL_SHA)):
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Bundle source identity check failed. No files were imported.")
        checkpoint = read_member(archive, "deploy.pt", 128 * 1024**2)
        if hashlib.sha256(checkpoint).hexdigest() != EXPECTED:
            raise ValueError("Unexpected trained checkpoint. No files were imported.")
        provenance = json.loads(read_member(archive, "cpu_parity.json", 512 * 1024))
        if provenance.get("complete") is not True or provenance.get("gpu_used") is not False:
            raise ValueError("The bundle must contain completed CPU parity evidence.")
        for name, expected in (("deploy.pt", EXPECTED), ("raw_preprocess.py", RAW_SHA),
                               ("cnn_convnext.py", ORIGINAL_MODEL_SHA)):
            if provenance.get("bundle_hashes", {}).get(name) != expected:
                raise ValueError("Bundle provenance does not bind the expected artifacts.")
    return checkpoint, provenance


def atomic_write(destination, content):
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".import-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def import_bundle(archive_path, root=ROOT):
    checkpoint, provenance = validate_bundle(archive_path, root)
    atomic_write(root / "models" / "original_cpu_parity.json", json.dumps(provenance, indent=2).encode())
    atomic_write(root / "models" / "deploy.pt", checkpoint)
    return {"checkpoint_sha256": EXPECTED, "application_source_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Already downloaded, authorized private Kaggle output ZIP")
    args = parser.parse_args()
    print(json.dumps(import_bundle(args.archive), indent=2))
