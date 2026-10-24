"""Bounded, fail-closed study archive admission. No model or network calls."""
from pathlib import Path, PurePosixPath
import csv
import os
import re
import shutil
import stat
import zipfile

UID = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
REQUIRED = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"}


def native_path(path):
    """Support full DICOM UID paths on Windows without changing system settings."""
    value = str(Path(path).resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def read_metadata(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("Series metadata must be smaller than 1 MB.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError("Metadata needs the five required series columns; see the README.")
        rows = list(reader)
    if not rows or len(rows) > 100:
        raise ValueError("Provide metadata for 1 to 100 series in one study.")
    studies, series = set(), set()
    for row in rows:
        study, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
        if any(not UID.fullmatch(x) or len(x) > 64 for x in (study, sid)):
            raise ValueError("Study and series identifiers must be valid numeric DICOM UIDs.")
        if sid in series:
            raise ValueError("Duplicate series identifier in metadata.")
        if row["Anatomical_Plane"].strip().title() not in {"Sagittal", "Coronal", "Axial"}:
            raise ValueError("Unsupported anatomical plane in metadata.")
        for name in ("Fat_Suppression", "Fluid_Sensitive"):
            if row[name].strip().lower() not in {"0", "1", "true", "false"}:
                raise ValueError("Sequence flags must be 0/1 or true/false.")
        studies.add(study)
        series.add(sid)
    if len(studies) != 1:
        raise ValueError("Upload exactly one study at a time.")
    return next(iter(studies)), series, rows


def extract_study(archive, metadata, destination, settings):
    destination = native_path(destination)
    study, allowed_series, rows = read_metadata(metadata)
    if archive.stat().st_size > settings.max_upload_bytes:
        raise ValueError("Study ZIP exceeds the 150 MB upload limit.")
    if not zipfile.is_zipfile(archive):
        raise ValueError("The study must be a valid ZIP archive.")
    destination.mkdir(parents=True, exist_ok=True)
    selected, seen, total, actual_series = [], set(), 0, set()
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist()) > settings.max_files + 102:
            raise ValueError("Too many archive entries.")
        for info in z.infolist():
            name = info.filename
            p = PurePosixPath(name)
            if "\\" in name or ":" in name or p.is_absolute() or any(x in {"..", "."} for x in p.parts):
                raise ValueError("Unsafe path inside the ZIP archive.")
            if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("Symlinks and encrypted ZIP entries are unsupported.")
            if info.is_dir():
                continue
            if len(p.parts) != 3 or p.parts[0] != study or p.parts[1] not in allowed_series or p.suffix.lower() != ".dcm":
                raise ValueError("ZIP layout must be StudyUID/SeriesUID/slice.dcm matching the metadata.")
            if name.casefold() in seen:
                raise ValueError("Duplicate file path in ZIP archive.")
            seen.add(name.casefold())
            total += info.file_size
            if info.file_size > settings.max_file_bytes or total > settings.max_expanded_bytes:
                raise ValueError("Expanded study exceeds the configured file or storage limit.")
            if info.compress_size and info.file_size / info.compress_size > 1000:
                raise ValueError("Archive compression ratio exceeds the allowed limit.")
            selected.append((info, p))
            actual_series.add(p.parts[1])
        if not selected or len(selected) > settings.max_files:
            raise ValueError("Archive must contain between 1 and 4000 DICOM slices.")
        if actual_series != allowed_series:
            raise ValueError("Some metadata series have no matching DICOM files.")
        root = destination.resolve()
        for info, p in selected:
            target = (destination / Path(*p.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escaped the job directory.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    # Inspect only needed headers; never copy patient fields into the result.
    import pydicom
    pixels = 0
    for path in (destination / study).glob("*/*.dcm"):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(ds.get("StudyInstanceUID", "")) != study or str(ds.get("SeriesInstanceUID", "")) != path.parent.name:
                raise ValueError("DICOM header identifiers do not match the archive and metadata.")
            if str(ds.get("Modality", "")) != "MR" or int(ds.get("NumberOfFrames", 1)) != 1:
                raise ValueError("Only single-frame MR DICOM slices are supported.")
            height, width = int(ds.get("Rows", 0)), int(ds.get("Columns", 0))
            if not 1 <= height <= 1024 or not 1 <= width <= 1024:
                raise ValueError("Image dimensions must be between 1 and 1024 pixels per axis.")
            pixels += height * width
            if pixels > settings.max_pixels:
                raise ValueError("Study pixel count exceeds the memory budget.")
        except (AttributeError, KeyError, TypeError, OSError):
            raise ValueError("A DICOM header is unreadable or incomplete.") from None
    return study, rows, len(selected)
