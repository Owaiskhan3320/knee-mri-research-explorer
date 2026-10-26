import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.check_release import check_release
from scripts.verify_training_archive import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archive_root(tmp_path):
    shutil.copytree(ROOT / "training", tmp_path / "training")
    for relative in ("app/model.py", "models/manifest.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_archive_verifies_without_data_or_weights(archive_root):
    assert not (archive_root / "models" / "deploy.pt").exists()
    assert not (archive_root / "data").exists()
    report = verify(archive_root)
    assert report["status"] == "TRAINING_ARCHIVE_PROVENANCE_PASS"
    assert report["notebook_training_enabled"] is False


def test_modified_historical_source_is_rejected(archive_root):
    path = archive_root / "training/archive/convnext/convnext_train.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Archived source changed"):
        verify(archive_root)


def test_enabled_notebook_training_is_rejected(archive_root):
    path = archive_root / "training/notebooks/convnext_tiny_budget.ipynb"
    notebook = json.loads(path.read_text())
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    cell["source"] = ["".join(cell["source"]).replace("RUN_GPU_TRAINING = False", "RUN_GPU_TRAINING = True", 1)]
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="RUN_GPU_TRAINING must be False"):
        verify(archive_root)


def test_release_rejects_unreviewed_training_outputs(tmp_path, monkeypatch):
    path = tmp_path / "training/private_predictions.csv"
    path.parent.mkdir()
    path.write_text("synthetic fixture only\n")
    monkeypatch.setattr("scripts.check_release.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(stdout=b"training/private_predictions.csv\0"))
    with pytest.raises(ValueError, match="Not in source/media allowlist"):
        check_release(tmp_path)
