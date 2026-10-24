import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from app.constants import CHECKPOINT_SHA256, TARGETS
from app.fixtures import create_fixture
from app.inputs import extract_study, native_path
from app.settings import ROOT, Settings
from app.worker import load_engine
from app import raw_preprocess as raw


@pytest.mark.requires_model
def test_checkpoint_and_preprocessor_identity():
    assert hashlib.sha256((ROOT / "models/deploy.pt").read_bytes()).hexdigest() == CHECKPOINT_SHA256
    assert hashlib.sha256((ROOT / "app/raw_preprocess.py").read_bytes()).hexdigest() == "69c185efb9f060d63463434c3e372e7a2c3a668dc33fbf7a77ccf13f589b01ed"


def test_compressed_dicom_decoders_are_available():
    from pydicom.pixels import get_decoder
    for uid in ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.90"):
        assert get_decoder(uid).is_available


@pytest.mark.requires_model
def test_real_model_masking_and_request_independence(tmp_path):
    model = load_engine(Settings(root=ROOT))
    archive, metadata = create_fixture(tmp_path / "input")
    uid, rows, _ = extract_study(archive, metadata, tmp_path / "data", Settings(root=tmp_path))
    x, valid, details = raw.load_study(uid, raw.group_series(pd.DataFrame(rows)), native_path(tmp_path / "data"))
    assert x.shape == (40, 3, 224, 224) and valid.all()
    images = torch.from_numpy(x[None])
    mask = torch.from_numpy(valid[None])
    slots = torch.arange(5).repeat_interleave(8)[None]
    with torch.inference_mode():
        first = model(images, mask, slots, chunk=8)
        mask[:, -8:] = False
        missing = model(images, mask, slots, chunk=8)
        changed = images.clone()
        changed[:, -8:] = 255
        masked_changed = model(changed, mask, slots, chunk=8)
        mask[:] = True
        repeated = model(images, mask, slots, chunk=8)
    assert first.shape == (1, 12) and torch.isfinite(first).all()
    assert torch.equal(first, repeated), "Another request changed the study result"
    assert torch.equal(missing, masked_changed), "Masked windows affected predictions"
