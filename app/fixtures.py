"""Self-generated synthetic DICOM fixture. Contains no competition or patient data."""
from pathlib import Path
import csv
import zipfile
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def create_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    archive, metadata = directory / "synthetic-study.zip", directory / "series_metadata.csv"
    study = generate_uid()
    rows = []
    yy, xx = np.mgrid[-1:1:96j, -1:1:96j]
    rng = np.random.default_rng(42)
    slots = [("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Sagittal", 0), ("Coronal", 0)]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for slot, (plane, fs) in enumerate(slots):
            series = generate_uid()
            rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=series, Anatomical_Plane=plane,
                             Fat_Suppression=fs, Fluid_Sensitive=1))
            for i in range(9):
                uid = generate_uid()
                fm = FileMetaDataset()
                fm.TransferSyntaxUID = ExplicitVRLittleEndian
                fm.MediaStorageSOPClassUID = MRImageStorage
                fm.MediaStorageSOPInstanceUID = uid
                ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
                ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, uid
                ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
                ds.Modality, ds.PatientName, ds.PatientID = "MR", "SYNTHETIC^PHANTOM", "SYNTHETIC"
                ds.Rows = ds.Columns = 96
                ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
                ds.BitsAllocated = ds.BitsStored = 16
                ds.HighBit, ds.PixelRepresentation = 15, 0
                ds.PixelSpacing, ds.InstanceNumber = [1.5, 1.5], i + 1
                ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [0, 0, i * 3.5]
                ds.RescaleSlope, ds.RescaleIntercept = 1, 0
                a = 80 + 900 * np.exp(-((xx - .22)**2 + (yy + .12)**2) * (5 + slot))
                a += (450 + i * 20) * np.exp(-((xx + .35)**2 + (yy - .25)**2) * 18)
                a += 20 * (xx + 1) + rng.normal(0, 4, xx.shape)
                ds.PixelData = np.clip(a, 0, 4095).astype("<u2").tobytes()
                import io
                f = io.BytesIO()
                ds.save_as(f, enforce_file_format=True)
                z.writestr(f"{study}/{series}/{uid}.dcm", f.getvalue())
    with metadata.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return archive, metadata
