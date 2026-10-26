"""Audit Git's publishable file set before staging or pushing. No network."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {".gitignore", ".gitattributes", ".dockerignore", "README.md", "LICENSE", "THIRD_PARTY.md", "Dockerfile",
              "requirements.txt", "requirements-lock.txt", "pyproject.toml", "run.py", "setup.ps1", "start.ps1"}
PUBLIC_MEDIA = {"docs/local-demo.png", "docs/demo.mp4"}
TRAINING_FILES = {
    "training/README.md", "training/provenance.json", "training/config/epoch8_config.json",
    "training/notebooks/convnext_tiny_budget.ipynb",
    "training/archive/convnext/cnn_convnext.py", "training/archive/convnext/parent_cnn_training.py",
    "training/archive/convnext/convnext_train.py", "training/archive/convnext/prepare_convnext.py",
    "training/archive/parent_preparation/cnn_training.py", "training/archive/parent_preparation/prepare.py",
    "training/archive/parent_preparation/label_audit.py",
}


def check_release(root=ROOT):
    command = ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    result = subprocess.run(command, check=True, capture_output=True)
    names = sorted(set(p.decode("utf-8") for p in result.stdout.split(b"\0") if p))
    if not names:
        raise ValueError("No release files found.")
    issues, files = [], []
    secrets = [re.compile(p) for p in (r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
               r"AKIA[A-Z0-9]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            issues.append(f"Missing file or symlink: {name}")
            continue
        allowed = name in ROOT_FILES or name in PUBLIC_MEDIA or name in TRAINING_FILES
        allowed |= name.startswith(("app/", "scripts/", "tests/")) and path.suffix == ".py"
        allowed |= name.startswith("docs/") and path.suffix in {".md", ".json", ".vtt"}
        allowed |= name in {"models/README.md", "models/manifest.json"}
        if not allowed:
            issues.append(f"Not in source/media allowlist: {name}")
            continue
        content = path.read_bytes()
        if name not in PUBLIC_MEDIA:
            text = content.decode("utf-8-sig")
            if any(pattern.search(text) for pattern in secrets):
                issues.append(f"Potential credential in: {name}")
            # Dataset UID values must not leak from private replay logs.
            if re.search(r"1\.2\.826\.0\.1\.3680043\.8\.498\.\d{10,}", text):
                issues.append(f"Real study/series identifier in: {name}")
        files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    if issues:
        raise ValueError("Release audit failed:\n" + "\n".join(issues))
    return {"status": "SOURCE_RELEASE_AUDIT_PASS", "files": files,
            "total_bytes": sum(f["bytes"] for f in files),
            "notice": "Media require visual review; an allowlisted name alone does not prove synthetic content."}


if __name__ == "__main__":
    try:
        report = check_release()
    except (ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
    print(json.dumps(report, indent=2))
