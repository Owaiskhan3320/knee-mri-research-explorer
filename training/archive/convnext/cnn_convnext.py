"""ConvNeXt-Tiny MIL and verified access to the unchanged V1 cache runtime.

Importing this file never downloads weights, trains, or uses a GPU. The official
ImageNet initialization is materialized by the separate CPU preparation notebook.
"""
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny


DEFAULT_PARENT_BUNDLE = "/kaggle/input/notebooks/owaiskhan15/knee-cnn-cpu-preparation/knee_cnn_preparation/bundle"
PARENT_MODULE_SHA256 = "fe38ece1a0a6e692d45f20e1f114cc876cbda16f97aec09c840e3adbc69a6ca2"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_parent():
    path = Path(__file__).with_name("parent_cnn_training.py")
    assert sha256(path) == PARENT_MODULE_SHA256, "V1 reader/runtime module changed"
    name = "knee_verified_v1_runtime_" + PARENT_MODULE_SHA256[:12]
    if name in sys.modules:
        module = sys.modules[name]
        assert sha256(module.__file__) == PARENT_MODULE_SHA256
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _import_parent()
PixelBags = base.PixelBags
loss_parts = base.loss_parts
augment = base.augment
metrics = base.metrics
predict = base.predict
atomic_torch = base.atomic_torch
write_json = base.write_json
fingerprint = base.fingerprint


class StudyMIL(nn.Module):
    """Trainable ConvNeXt-Tiny encoder with finding-specific attention pooling."""
    def __init__(self, num_targets=12, slots=5, projection=256,
                 encoder_state=None, pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError("Use the official initialization from CPU preparation")
        self.encoder = convnext_tiny(weights=None)
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state, strict=True)
        # Preserve torchvision's final LayerNorm and flatten before its classifier.
        self.encoder.classifier[2] = nn.Identity()
        self.project = nn.Sequential(nn.Linear(768, projection), nn.LayerNorm(projection),
                                     nn.SiLU(), nn.Dropout(0.2))
        self.slot = nn.Embedding(slots, projection)
        self.av = nn.Linear(projection, 128)
        self.au = nn.Linear(projection, 128)
        self.attention = nn.Linear(128, num_targets)
        self.finding_weight = nn.Parameter(torch.empty(num_targets, projection))
        self.finding_bias = nn.Parameter(torch.zeros(num_targets))
        nn.init.xavier_uniform_(self.finding_weight)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images, valid, slots, chunk=32):
        assert images.ndim == 5 and images.shape[2] == 3
        b, w, c, h, width = images.shape
        assert valid.shape == slots.shape == (b, w)
        assert valid.dtype == torch.bool and slots.dtype == torch.long
        assert chunk >= 1 and bool(valid.any(1).all()), "Empty study bag"
        x = images.reshape(b * w, c, h, width)
        x = x.float().div(255) if not x.is_floating_point() else x
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        x = ((x[indices] - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        encoded = torch.cat([self.encoder(part) for part in x.split(chunk)], 0)
        assert encoded.shape == (len(indices), 768)
        features = encoded.new_zeros((b * w, 768)).index_copy(0, indices, encoded)
        tokens = self.project(features.reshape(b, w, 768)) + self.slot(slots)
        scores = self.attention(torch.tanh(self.av(tokens)) * torch.sigmoid(self.au(tokens)))
        scores = scores.transpose(1, 2).float().masked_fill(~valid[:, None, :], -torch.inf)
        attention = scores.softmax(-1).to(tokens.dtype)
        pooled = torch.einsum("bcw,bwd->bcd", attention, tokens)
        return (pooled * self.finding_weight[None]).sum(-1) + self.finding_bias


def load_bundle(bundle_path, require_preflight=False):
    root = Path(bundle_path)
    cfg = json.loads((root / "config.json").read_text())
    assert cfg["encoder"] == "torchvision.convnext_tiny"
    assert cfg["parent_module_sha256"] == PARENT_MODULE_SHA256
    required = {"cnn_convnext.py", "parent_cnn_training.py", "prepare_convnext.py", "convnext_train.py",
                "study_labels_splits.csv", "pixel_index.csv", "initial_model.pt"}
    assert required <= set(cfg["hashes"]), "Incomplete ConvNeXt bundle hashes"
    assert sha256(__file__) == cfg["hashes"]["cnn_convnext.py"]
    for name, expected in cfg["hashes"].items():
        path = (root / name).resolve()
        assert root.resolve() in path.parents, "Bundle path escaped root"
        assert sha256(path) == expected, "Bundle file changed: " + name
    parent_root, parent_cfg, parent_frame, parent_pixels = base.load_bundle(
        cfg["parent_bundle"], require_preflight=True)
    assert sha256(parent_root / "config.json") == cfg["parent_config_sha256"]
    assert cfg["targets"] == parent_cfg["targets"] and cfg["slots"] == parent_cfg["slots"]
    assert cfg["pixel_key"] == parent_cfg["pixel_key"]
    assert cfg["data_hashes"] == parent_cfg["data_hashes"]
    for name in ["study_labels_splits.csv", "pixel_index.csv"]:
        assert sha256(root / name) == sha256(parent_root / name), "Original data changed"
    if require_preflight:
        assert "preflight.json" in cfg["hashes"]
        report = json.loads((root / "preflight.json").read_text())
        assert report["complete"] and report["gpu_used"] is False
        assert report["data_fingerprint"] == fingerprint(cfg)
        assert report["initial_state_sha256"] == cfg["hashes"]["initial_model.pt"]
        assert report["encoder_backward_verified"] and report["missing_mask_verified"]
    return root, cfg, parent_frame, parent_pixels


def model_from_bundle(bundle_path, require_preflight=True):
    root, cfg, _, _ = load_bundle(bundle_path, require_preflight=require_preflight)
    model = StudyMIL(len(cfg["targets"]), len(cfg["slots"]), cfg["settings"]["projection"])
    model.load_state_dict(torch.load(root / "initial_model.pt", map_location="cpu",
                                    weights_only=True), strict=True)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    return model


def cpu_preflight(bundle_path):
    """Verify backward/missing masks on two valid windows, then full-bag inference.

    This is a correctness check, not a benchmark of training speed or memory for
    the eventual full training batch. No optimizer update is performed.
    """
    assert not torch.cuda.is_available(), "CPU preparation requires accelerator None"
    root, cfg, frame, pixels = load_bundle(bundle_path)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = model_from_bundle(root, require_preflight=False).cpu().train()
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.encoder.modules())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    train = PixelBags(frame.loc[frame.split.eq("fit")].iloc[:1], pixels,
                      cfg["targets"], cfg["pixel_key"], training=True, seed=cfg["seed"],
                      windows_per_slot=2)
    batch = next(iter(DataLoader(train, batch_size=1, num_workers=0)))
    available = batch["valid"][0].nonzero(as_tuple=True)[0]
    assert len(available) >= 2
    chosen = available[[0, -1]]
    # One deliberately invalid token has a real image payload, so a leaking mask
    # would affect the output instead of passing accidentally on all-zero padding.
    images = torch.cat([batch["images"][:, chosen], batch["images"][:, chosen[:1]]], dim=1)
    valid = torch.tensor([[True, True, False]])
    slots = torch.cat([batch["slots"][:, chosen], batch["slots"][:, chosen[:1]]], dim=1)
    output = model(images, valid, slots, chunk=2)
    numerator, count = loss_parts(output, batch)
    assert int(count) > 0 and torch.isfinite(numerator) and output.shape == (1, 12)
    loss = numerator / count
    loss.backward()
    gradients = [parameter.grad for parameter in model.encoder.parameters()]
    assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
    assert any(bool(value.abs().sum() > 0) for value in gradients)
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    del before, output, numerator, gradients
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        original = model(images, valid, slots, chunk=2)
        changed = images.clone()
        changed[:, 2] = 255 - changed[:, 2]
        masked = model(changed, valid, slots, chunk=2)
        unpadded = model(images[:, :2], valid[:, :2], slots[:, :2], chunk=2)
        order = torch.tensor([2, 1, 0])
        permuted = model(images[:, order], valid[:, order], slots[:, order], chunk=2)
        assert torch.allclose(original, masked, atol=1e-6, rtol=1e-6)
        assert torch.allclose(original, unpadded, atol=1e-6, rtol=1e-6)
        assert torch.allclose(original, permuted, atol=1e-5, rtol=1e-5)
    val = PixelBags(frame.loc[frame.split.eq("weak_validation")].iloc[:1], pixels,
                    cfg["targets"], cfg["pixel_key"])
    vb = next(iter(DataLoader(val, batch_size=1, num_workers=0)))
    assert vb["images"].shape == (1, 40, 3, 224, 224)
    with torch.inference_mode():
        probabilities = model(vb["images"], vb["valid"], vb["slots"], chunk=2).sigmoid()
    assert probabilities.shape == (1, 12) and torch.isfinite(probabilities).all()
    report = dict(complete=True, gpu_used=False, data_fingerprint=fingerprint(cfg),
                  initial_state_sha256=cfg["hashes"]["initial_model.pt"],
                  targets=cfg["targets"], training_loss=float(loss.detach()),
                  backward_valid_windows=2, backward_invalid_windows=1,
                  validation_bag_windows=40, validation_valid_windows=int(vb["valid"].sum()),
                  encoder_backward_verified=True, initial_state_unchanged=True,
                  full_mil_gradients_finite=True,
                  missing_mask_verified=True, permutation_invariance_verified=True,
                  sampled_uids=batch["uid"] + vb["uid"],
                  limitation="Two-valid-window CPU backward is not a full training-batch memory benchmark")
    del model, batch, vb, images
    gc.collect()
    return report
