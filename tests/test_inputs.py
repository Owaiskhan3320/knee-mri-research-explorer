from dataclasses import replace
import csv
from pathlib import Path
import zipfile
import pytest
from app.inputs import extract_study, native_path
from app.fixtures import create_fixture
from app.settings import Settings


def test_actual_synthetic_dicom_is_admitted(tmp_path):
    archive, metadata = create_fixture(tmp_path / "input")
    uid, rows, count = extract_study(archive, metadata, tmp_path / "extracted", Settings(root=tmp_path))
    assert count == 45 and len(rows) == 5
    assert len(list(native_path(tmp_path / "extracted" / uid).rglob("*.dcm"))) == 45


@pytest.mark.parametrize("bad", ["../escape.dcm", "/absolute.dcm", "C:/escape.dcm", "a\\b\\c.dcm"])
def test_archive_paths_cannot_escape(tmp_path, bad):
    archive, metadata = create_fixture(tmp_path / "input")
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr(bad, b"bad")
    with pytest.raises(ValueError):
        extract_study(archive, metadata, tmp_path / "extracted", Settings(root=tmp_path))
    assert not (tmp_path / "escape.dcm").exists()


def test_expansion_limit_applies_before_extraction(tmp_path):
    archive, metadata = create_fixture(tmp_path / "input")
    with pytest.raises(ValueError, match="Expanded"):
        extract_study(archive, metadata, tmp_path / "extracted", replace(Settings(root=tmp_path), max_expanded_bytes=1))
    assert not list((tmp_path / "extracted").rglob("*.dcm"))


def test_metadata_cannot_reassign_a_study(tmp_path):
    archive, metadata = create_fixture(tmp_path / "input")
    with metadata.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["StudyInstanceUID"] = "1.2.3"
    with metadata.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="exactly one study"):
        extract_study(archive, metadata, tmp_path / "extracted", Settings(root=tmp_path))


def test_pixel_budget_is_enforced(tmp_path):
    archive, metadata = create_fixture(tmp_path / "input")
    with pytest.raises(ValueError, match="pixel count"):
        extract_study(archive, metadata, tmp_path / "extracted", replace(Settings(root=tmp_path), max_pixels=1))
