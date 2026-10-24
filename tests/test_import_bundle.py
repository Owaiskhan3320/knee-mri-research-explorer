"""Malformed imports must leave existing model assets and source intact."""
import shutil
import zipfile
import pytest
from app.settings import ROOT
from scripts.import_bundle import PREFIX, import_bundle


@pytest.mark.parametrize("problem", ["changed-source", "duplicate-source"])
def test_rejected_bundle_preserves_existing_assets(tmp_path, problem):
    for directory in ("app", "models"):
        (tmp_path / directory).mkdir()
    for relative in ("app/raw_preprocess.py", "app/model.py", "models/manifest.json"):
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    (tmp_path / "models/deploy.pt").write_bytes(b"existing-private-model")
    archive = tmp_path / "untrusted.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        raw = (ROOT / "app/raw_preprocess.py").read_bytes()
        bundle.writestr(PREFIX + "raw_preprocess.py", raw + b"# changed" if problem == "changed-source" else raw)
        if problem == "duplicate-source":
            with pytest.warns(UserWarning, match="Duplicate"):
                bundle.writestr(PREFIX + "raw_preprocess.py", raw)
        bundle.writestr(PREFIX + "cnn_convnext.py", b"untrusted executable source")
        bundle.writestr(PREFIX + "deploy.pt", b"wrong checkpoint")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p != archive}
    with pytest.raises(ValueError):
        import_bundle(archive, tmp_path)
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p != archive}
    assert after == before
