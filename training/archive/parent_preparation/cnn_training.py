"""Offline, budget-limited EfficientNet-B0 study MIL. Importing never starts work."""
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0


DEFAULTS = dict(seed=42, batch_size=2, accumulation=4, epochs=8, workers=2,
                encoder_lr=3e-5, head_lr=3e-4, weight_decay=1e-4,
                train_windows_per_slot=2, projection=256, patience=3,
                min_delta=1e-4, warmup_steps=40, encoder_chunk=32, intensity_jitter=.1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def fingerprint(cfg):
    clean = {k: v for k, v in cfg.items() if k != "hashes"}
    clean["hashes"] = {k: v for k, v in cfg["hashes"].items()
                       if k != "preflight.json"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def load_bundle(bundle_path, require_preflight=False):
    root = Path(bundle_path)
    cfg = json.loads((root / "config.json").read_text())
    assert len(cfg["targets"]) == 12 and len(set(cfg["targets"])) == 12
    for required in ("study_labels_splits.csv", "pixel_index.csv", "initial_model.pt",
                     "cnn_training.py"):
        assert required in cfg["hashes"], f"Missing required hash: {required}"
    assert sha256(Path(__file__)) == cfg["hashes"]["cnn_training.py"], "Executed module changed"
    for name, expected in cfg["hashes"].items():
        path = (root / name).resolve()
        assert root.resolve() in path.parents, f"Bundle path escaped root: {name}"
        assert sha256(path) == expected, f"Bundle hash changed: {name}"
    for path, expected in cfg.get("source_hashes", {}).items():
        assert sha256(path) == expected, f"Parent data changed: {path}"
    frame = pd.read_csv(root / "study_labels_splits.csv", dtype={"StudyInstanceUID": str})
    pixels = pd.read_csv(root / "pixel_index.csv", dtype={"StudyInstanceUID": str})
    uid = "StudyInstanceUID"
    assert frame[uid].notna().all() and frame[uid].is_unique and pixels[uid].is_unique
    assert set(frame["split"]) == {"fit", "weak_validation", "gold"}
    assert set(frame[uid]) <= set(pixels[uid])
    assert {uid, "root", "shard", "member"} <= set(pixels.columns)
    assert all(t in frame and t + "__binary" in frame for t in cfg["targets"])
    for target in cfg["targets"]:
        soft = frame[target].dropna().to_numpy(float)
        binary = frame[target + "__binary"].dropna().to_numpy(float)
        assert np.isfinite(soft).all() and ((soft >= 0) & (soft <= 1)).all()
        assert np.isin(binary, [0, 1]).all()
        assert np.array_equal(frame[target].notna(), frame[target + "__binary"].notna())
    if "verified_group" in frame:
        assert frame.verified_group.notna().all()
        assert frame.groupby("verified_group")["split"].nunique().max() == 1
    else:
        raise AssertionError("Patient/report verified_group is required")
    counts = frame["split"].value_counts().to_dict()
    for split, expected in cfg.get("expected_counts", {}).items():
        assert counts.get(split) == expected, (split, counts)
    if require_preflight:
        assert "preflight.json" in cfg["hashes"], "CPU preflight must be bundled and hashed"
        report = json.loads((root / "preflight.json").read_text())
        assert report["complete"] is True and report["gpu_used"] is False
        assert report["data_fingerprint"] == fingerprint(cfg)
        assert report["initial_state_sha256"] == cfg["hashes"]["initial_model.pt"]
    return root, cfg, frame, pixels


class StudyMIL(nn.Module):
    """Trainable B0 encoder, frozen BN statistics, and finding-specific attention."""
    def __init__(self, num_targets=12, slots=5, projection=256,
                 encoder_state=None, pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError("Download official weights in CPU preparation; pass encoder_state")
        self.encoder = efficientnet_b0(weights=None)
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.classifier = nn.Identity()
        self.project = nn.Sequential(nn.Linear(1280, projection), nn.LayerNorm(projection),
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

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images, valid, slots, chunk=32):
        b, w, c, h, width = images.shape
        x = images.reshape(b * w, c, h, width)
        x = x.float().div(255) if not x.is_floating_point() else x
        flat_valid = valid.reshape(-1)
        indices = flat_valid.nonzero(as_tuple=True)[0]
        assert indices.numel() > 0 and bool(valid.any(1).all()), "Empty study bag"
        x = ((x[indices] - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        encoded = torch.cat([self.encoder(part) for part in x.split(chunk)], 0)
        features = encoded.new_zeros((b * w, 1280)).index_copy(0, indices, encoded)
        tokens = self.project(features.reshape(b, w, 1280)) + self.slot(slots)
        scores = self.attention(torch.tanh(self.av(tokens)) * torch.sigmoid(self.au(tokens)))
        scores = scores.transpose(1, 2).float().masked_fill(~valid[:, None, :], -torch.inf)
        attention = scores.softmax(-1).to(tokens.dtype)
        pooled = torch.einsum("bcw,bwd->bcd", attention, tokens)
        return (pooled * self.finding_weight[None]).sum(-1) + self.finding_bias


class PixelBags(Dataset):
    """Packed-cache reader. Epoch is carried in each sampler index, never worker state."""
    def __init__(self, frame, pixel_index, targets, pixel_key, training=False,
                 seed=42, windows_per_slot=2):
        self.frame = frame.reset_index(drop=True).copy()
        self.records = pixel_index.set_index("StudyInstanceUID").to_dict("index")
        self.targets, self.pixel_key = list(targets), pixel_key
        self.training, self.seed, self.windows_per_slot = training, seed, windows_per_slot
        assert 1 <= windows_per_slot <= 8
        self._pid, self._zips, self._verified = None, OrderedDict(), set()

    def __len__(self):
        return len(self.frame)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _zips=OrderedDict(), _verified=set())
        return state

    def _read(self, uid):
        if self._pid != os.getpid():
            for archive in self._zips.values():
                archive.close()
            self._pid, self._zips, self._verified = os.getpid(), OrderedDict(), set()
        row = self.records[uid]
        path = str(Path(row["root"]) / "pixels" / row["shard"])
        if path not in self._zips:
            self._zips[path] = zipfile.ZipFile(path)
        archive = self._zips.pop(path)
        self._zips[path] = archive
        while len(self._zips) > 8:
            self._zips.popitem(last=False)[1].close()
        data = archive.read(row["member"])
        expected = row.get("packed_sha256")
        if uid not in self._verified and isinstance(expected, str) and expected:
            assert hashlib.sha256(data).hexdigest() == expected, f"Bad pixels: {uid}"
            self._verified.add(uid)
        return data

    def __getitem__(self, index):
        epoch, position = index if isinstance(index, tuple) else (0, index)
        row = self.frame.iloc[int(position)]
        uid = str(row.StudyInstanceUID)
        with np.load(io.BytesIO(self._read(uid)), allow_pickle=False) as packed:
            assert str(packed["uid"].item()) == uid
            assert str(packed["key"].item()) == self.pixel_key
            valid = packed["valid"].astype(bool)
            refs = packed["refs"]
            assert valid.shape == (40,) and refs.shape == (40, 3) and valid.any()
            original_valid = valid.copy()
            if self.training:
                stable = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4], "little")
                rng = np.random.default_rng([self.seed, int(epoch), stable])
                selected, selected_valid = [], []
                for slot in range(5):
                    candidates = np.flatnonzero(valid[slot * 8:(slot + 1) * 8]) + slot * 8
                    n = min(len(candidates), self.windows_per_slot)
                    choices = rng.choice(candidates, n, replace=False).tolist() if n else []
                    selected += choices + [slot * 8] * (self.windows_per_slot - n)
                    selected_valid += [True] * n + [False] * (self.windows_per_slot - n)
                valid = np.asarray(selected_valid, dtype=bool)
                slots = np.repeat(np.arange(5), self.windows_per_slot)
            else:
                selected, slots = np.arange(40), np.repeat(np.arange(5), 8)
            frames = packed["frames"]
            assert frames.dtype == np.uint8 and frames.shape[1:] == (224, 224)
            assert refs[original_valid].min() >= 0 and refs[original_valid].max() < len(frames)
            selected_refs = refs[selected].copy()
            selected_refs[~valid] = 0
            images = frames[selected_refs].copy()
            images[~valid] = 0
        y = row[self.targets].to_numpy(dtype=np.float32)
        binary = row[[t + "__binary" for t in self.targets]].to_numpy(dtype=np.float32)
        return dict(images=torch.from_numpy(images), valid=torch.from_numpy(valid.copy()),
                    slots=torch.as_tensor(slots, dtype=torch.long),
                    y=torch.from_numpy(np.nan_to_num(y)), mask=torch.from_numpy(np.isfinite(y)),
                    binary=torch.from_numpy(np.nan_to_num(binary)),
                    binary_mask=torch.from_numpy(np.isfinite(binary)), uid=uid)


def loss_parts(logits, batch):
    mask = batch["mask"]
    safe = torch.where(mask, batch["y"], torch.zeros_like(batch["y"]))
    losses = F.binary_cross_entropy_with_logits(logits.float(), safe.float(), reduction="none")
    return losses.masked_select(mask).sum(), mask.sum()


def augment(images, intensity_jitter=.1):
    x = images.float().div(255)
    gain = 1 + intensity_jitter * (2 * torch.rand(len(x), 1, 1, 1, 1, device=x.device) - 1)
    return (x * gain).clamp(0, 1)


def _settings(cfg):
    settings = dict(DEFAULTS, **cfg.get("settings", {}))
    assert 1 <= settings["epochs"] <= 8 and 1 <= settings["batch_size"] <= 4
    assert 1 <= settings["accumulation"] <= 8 and 0 <= settings["workers"] <= 4
    assert 0 <= settings["intensity_jitter"] <= .2
    return settings


def _model(root, cfg, settings):
    model = StudyMIL(len(cfg["targets"]), projection=settings["projection"])
    state = torch.load(root / "initial_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    assert all(p.requires_grad for p in model.encoder.parameters())
    return model


def cpu_preflight(bundle_path):
    """One backward pass without optimizer updates; returns a JSON-safe report."""
    root, cfg, frame, pixels = load_bundle(bundle_path)
    settings = _settings(cfg)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = _model(root, cfg, settings).cpu().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit = frame[frame.split == "fit"].iloc[:2]
    dataset = PixelBags(fit, pixels, cfg["targets"], cfg["pixel_key"], True,
                        settings["seed"], settings["train_windows_per_slot"])
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    output = model(batch["images"], batch["valid"], batch["slots"])
    numerator, count = loss_parts(output, batch)
    assert int(count) > 0 and torch.isfinite(numerator)
    loss = numerator / count
    loss.backward()
    assert output.shape == (len(fit), 12)
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(bool(g.abs().sum() > 0) for g in grads)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(not m.training for m in model.encoder.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
    model.eval()
    val = PixelBags(frame[frame.split == "weak_validation"].iloc[:1], pixels,
                    cfg["targets"], cfg["pixel_key"])
    vb = next(iter(DataLoader(val, batch_size=1, num_workers=0)))
    with torch.inference_mode():
        vp = model(vb["images"], vb["valid"], vb["slots"]).sigmoid()
    assert vp.shape == (1, 12) and torch.isfinite(vp).all()
    return dict(complete=True, gpu_used=False, data_fingerprint=fingerprint(cfg),
                initial_state_sha256=cfg["hashes"]["initial_model.pt"],
                targets=cfg["targets"], training_loss=float(loss.detach()),
                train_windows=int(batch["valid"].sum()), validation_windows=int(vb["valid"].sum()),
                encoder_backward_verified=True, initial_state_unchanged=True,
                sampled_uids=batch["uid"] + vb["uid"])


def _device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def _loader(dataset, settings, batches=None):
    kwargs = dict(num_workers=settings["workers"], pin_memory=True,
                  persistent_workers=False)
    if settings["workers"]:
        kwargs["prefetch_factor"] = 2
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **kwargs)
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False, **kwargs)


def metrics(predictions, frame, targets):
    from sklearn.metrics import roc_auc_score
    aligned = frame.set_index("StudyInstanceUID").loc[predictions.StudyInstanceUID]
    rows = []
    for target in targets:
        y = aligned[target + "__binary"].to_numpy(float)
        p = predictions[target].to_numpy(float)
        known = np.isfinite(y)
        auc = roc_auc_score(y[known], p[known]) if len(np.unique(y[known])) == 2 else np.nan
        rows.append(dict(target=target, count=int(known.sum()),
                         positives=int(np.nansum(y)), auc=auc))
    table = pd.DataFrame(rows)
    return table, float(table.auc.mean())


def predict(model, dataset, settings, device, deadline=float("inf"), reserve=20, limit=None):
    model.eval()
    predictions, loss_sum, count, complete = [], 0., 0, True
    start = time.monotonic()
    with torch.inference_mode():
        for index, batch in enumerate(_loader(dataset, settings)):
            if time.monotonic() + reserve + 2 >= deadline:
                complete = False
                break
            batch = _device_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(batch["images"], batch["valid"], batch["slots"],
                               settings["encoder_chunk"])
            numerator, known = loss_parts(logits, batch)
            probabilities = logits.float().sigmoid().cpu().numpy()
            assert np.isfinite(probabilities).all()
            predictions.extend([[uid] + p.tolist() for uid, p in zip(batch["uid"], probabilities)])
            loss_sum += float(numerator)
            count += int(known)
            if limit is not None and index + 1 >= limit:
                complete = len(predictions) == len(dataset)
                break
    frame = pd.DataFrame(predictions, columns=["StudyInstanceUID"] + dataset.targets)
    return frame, loss_sum / max(count, 1), complete, time.monotonic() - start


def _rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def _restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def run_training(bundle_path, max_minutes=90, resume_path=None,
                 output_path="/kaggle/working/knee_effnet_b0"):
    """Explicit GPU entry point. Never submits; clean budget stops remain resumable."""
    started = time.monotonic()
    assert 1 <= max_minutes <= 90
    deadline = started + max_minutes * 60
    root, cfg, frame, pixels = load_bundle(bundle_path, require_preflight=True)
    settings = _settings(cfg)
    assert torch.cuda.is_available(), "CPU preflight must precede an explicitly enabled GPU run"
    device = torch.device("cuda:0")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True
    model = _model(root, cfg, settings).to(device, memory_format=torch.channels_last)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=list(model.encoder.parameters()), lr=settings["encoder_lr"]),
        dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],
             lr=settings["head_lr"])], weight_decay=settings["weight_decay"])
    train_frame = frame[frame.split == "fit"].reset_index(drop=True)
    val_frame = frame[frame.split == "weak_validation"].reset_index(drop=True)
    gold_frame = frame[frame.split == "gold"].reset_index(drop=True)
    train = PixelBags(train_frame, pixels, cfg["targets"], cfg["pixel_key"], True,
                      settings["seed"], settings["train_windows_per_slot"])
    val = PixelBags(val_frame, pixels, cfg["targets"], cfg["pixel_key"])
    gold = PixelBags(gold_frame, pixels, cfg["targets"], cfg["pixel_key"])
    epoch_batches = math.ceil(len(train) / settings["batch_size"])
    epoch_steps = math.ceil(epoch_batches / settings["accumulation"])
    planned_steps = settings["epochs"] * epoch_steps
    def lr_scale(step):
        if step < settings["warmup_steps"]:
            return (step + 1) / settings["warmup_steps"]
        position = min(1., (step - settings["warmup_steps"]) /
                       max(1, planned_steps - settings["warmup_steps"]))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * position))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda")
    state = dict(epoch=0, batch_cursor=0, order=None, phase="train", optimizer_steps=0,
                 completed_epochs=0, best_loss=float("inf"), best_epoch=None,
                 bad_epochs=0, history=[], best_model=None, best_predictions=None,
                 train_loss_sum=0., train_known=0, skipped_updates=0,
                 consecutive_overflows=0, reason="running")
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        assert saved["data_fingerprint"] == fingerprint(cfg)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state.update(saved["progress"])
        _restore_rng(saved["rng"])
    def checkpoint(path):
        atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                          progress=state, rng=_rng_state(), data_fingerprint=fingerprint(cfg),
                          config=cfg, gpu_name=torch.cuda.get_device_name(0)), path)
    checkpoint(out / "last.pt")
    group_times, val_estimate, bench_done = [], None, False
    benchmark_viable = True
    early_stop, budget_stop = state["bad_epochs"] >= settings["patience"], False
    while state["epoch"] < settings["epochs"] and not early_stop:
        epoch = state["epoch"]
        epoch_started = time.monotonic()
        if state["order"] is None:
            state["order"] = np.random.default_rng(settings["seed"] + epoch).permutation(len(train)).tolist()
        batches = [[(epoch, i) for i in state["order"][j:j + settings["batch_size"]]]
                   for j in range(0, len(train), settings["batch_size"])]
        gold_reserve = (val_estimate or 90.) * len(gold) / max(len(val), 1) * 1.3
        reserve = 20 + gold_reserve
        if state["batch_cursor"] == 0 and group_times and val_estimate is not None:
            estimate = 1.25 * (np.mean(group_times[-40:]) * epoch_steps + val_estimate)
            if time.monotonic() + estimate + reserve >= deadline:
                budget_stop = True
                break
        model.train()
        assert all(p.requires_grad for p in model.encoder.parameters())
        assert all(not m.training for m in model.encoder.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm))
        iterator = iter(_loader(train, settings, batches[state["batch_cursor"]:]))
        while state["batch_cursor"] < len(batches):
            expected = max(group_times[-10:] or [8.]) * 1.5
            training_reserve = reserve + 1.25 * (val_estimate or 0.)
            if time.monotonic() + expected + training_reserve >= deadline:
                budget_stop = True
                break
            group_start = time.monotonic()
            remaining = min(settings["accumulation"], len(batches) - state["batch_cursor"])
            group = [next(iterator) for _ in range(remaining)]
            denominator = sum(int(batch["mask"].sum()) for batch in group)
            assert denominator > 0
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = _device_batch(batch, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(augment(batch["images"], settings["intensity_jitter"]),
                                   batch["valid"], batch["slots"],
                                   settings["encoder_chunk"])
                numerator, known = loss_parts(logits, batch)
                assert torch.isfinite(numerator), "Nonfinite training loss"
                scaler.scale(numerator / denominator).backward()
                state["train_loss_sum"] += float(numerator.detach())
                state["train_known"] += int(known)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            state["batch_cursor"] += remaining
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                state["optimizer_steps"] += 1
                state["consecutive_overflows"] = 0
            else:
                state["skipped_updates"] += 1
                state["consecutive_overflows"] += 1
                print("AMP overflow: update skipped; scaler reduced.", flush=True)
                if state["consecutive_overflows"] >= 8:
                    checkpoint(out / "last.pt")
                    raise FloatingPointError("Eight consecutive AMP overflows; stopped without retrying")
            torch.cuda.synchronize()
            group_times.append(time.monotonic() - group_start)
            if state["optimizer_steps"] % 20 == 0:
                print(dict(epoch=epoch + 1, batch=state["batch_cursor"],
                           batches=len(batches), minutes=round((time.monotonic()-started)/60, 2)), flush=True)
            if state["optimizer_steps"] % 50 == 0:
                checkpoint(out / "last.pt")
            if not bench_done and len(group_times) >= 20:
                probe, _, _, elapsed = predict(model, val, settings, device, deadline, reserve, limit=8)
                assert len(probe) > 0, "No validation benchmark completed"
                val_estimate = elapsed * len(val) / len(probe)
                bench_done = True
                projection = 1.25 * (np.mean(group_times) * epoch_steps + val_estimate)
                print(dict(benchmark_train_seconds_per_step=float(np.mean(group_times)),
                           projected_epoch_minutes=projection / 60,
                           projected_validation_minutes=val_estimate / 60,
                           estimated_epochs_in_remaining_budget=max(0, int((deadline-time.monotonic()-reserve)/projection))), flush=True)
                write_json(dict(training_steps=len(group_times), seconds_per_step=float(np.mean(group_times)),
                                estimated_validation_seconds=val_estimate,
                                estimated_epoch_seconds=projection, max_minutes=max_minutes), out / "benchmark.json")
                model.train()
                remaining_steps = math.ceil((len(batches) - state["batch_cursor"]) /
                                            settings["accumulation"])
                finish_estimate = 1.25 * (np.mean(group_times) * remaining_steps + val_estimate)
                if time.monotonic() + finish_estimate + reserve >= deadline:
                    print("Benchmark cannot fit one complete epoch plus validation; stopping early.", flush=True)
                    benchmark_viable, budget_stop = False, True
                    break
        del iterator
        if budget_stop:
            break
        state["phase"] = "validation"
        predictions, val_loss, complete, elapsed = predict(model, val, settings, device, deadline, reserve)
        if not complete:
            predictions.to_csv(out / "val_predictions_partial.csv", index=False)
            budget_stop = True
            break
        assert len(predictions) == len(val) and predictions.StudyInstanceUID.is_unique
        val_estimate = elapsed
        auc_table, auc = metrics(predictions, val_frame, cfg["targets"])
        record = dict(epoch=epoch + 1, training_loss=state["train_loss_sum"] / max(state["train_known"], 1),
                      weak_validation_loss=val_loss, weak_macro_auc=auc,
                      epoch_minutes=(time.monotonic() - epoch_started) / 60)
        state["history"].append(record)
        print(record, flush=True)
        improved = val_loss < state["best_loss"] - settings["min_delta"]
        if improved:
            state.update(best_loss=val_loss, best_epoch=epoch + 1, bad_epochs=0,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         best_predictions=predictions.to_dict("list"))
        else:
            state["bad_epochs"] += 1
        state.update(epoch=epoch + 1, completed_epochs=epoch + 1, batch_cursor=0, order=None,
                     phase="train", train_loss_sum=0., train_known=0)
        pd.DataFrame(state["history"]).to_csv(out / "training_history.csv", index=False)
        if improved:
            checkpoint(out / "best.pt")
            auc_table.to_csv(out / "val_auc.csv", index=False)
        checkpoint(out / "last.pt")
        if state["bad_epochs"] >= settings["patience"]:
            early_stop = True
            break
    state["reason"] = "budget_stopped" if budget_stop else "early_stopped" if early_stop else "epoch_limit"
    checkpoint(out / "last.pt")
    summary = dict(complete=not budget_stop, candidate_ready=state["best_model"] is not None,
                   benchmark_viable=benchmark_viable, training_reason=state["reason"],
                   skipped_optimizer_updates=state["skipped_updates"],
                   planned_epochs=settings["epochs"], completed_epochs=state["completed_epochs"],
                   active_epoch=state["epoch"] + 1, next_batch=state["batch_cursor"],
                   best_epoch=state["best_epoch"], gpu_used=True, max_gpu_minutes=max_minutes,
                   gpu_name=torch.cuda.get_device_name(0), data_fingerprint=fingerprint(cfg),
                   expert_role="development diagnostic; never checkpoint/weight selection",
                   submission_created=False, selection_metric="masked soft-label weak-validation BCE")
    if state["best_model"] is not None:
        model.load_state_dict(state["best_model"], strict=True)
        selected = pd.DataFrame(state["best_predictions"])
        selected.to_csv(out / "val_predictions.csv", index=False)
        auc_table, auc = metrics(selected, val_frame, cfg["targets"])
        auc_table.to_csv(out / "val_auc.csv", index=False)
        summary.update(weak_validation_loss=state["best_loss"], weak_macro_auc=auc)
        atomic_torch(dict(model=state["best_model"], config=cfg, selected_epoch=state["best_epoch"],
                          trained=True, targets=cfg["targets"], data_fingerprint=fingerprint(cfg)), out / "deploy.pt")
        gp, _, complete, _ = predict(model, gold, settings, device, deadline, 20)
        summary["expert_evaluation_complete"] = complete
        if complete:
            gp.to_csv(out / "gold_predictions.csv", index=False)
            ga, gauc = metrics(gp, gold_frame, cfg["targets"])
            ga.to_csv(out / "gold_auc.csv", index=False)
            summary["expert_development_macro_auc"] = gauc
        else:
            gp.to_csv(out / "gold_predictions_partial.csv", index=False)
    summary["elapsed_gpu_job_minutes"] = (time.monotonic() - started) / 60
    write_json(summary, out / "run_summary.json")
    print("TRAINING_BUDGET_STOPPED" if budget_stop else "TRAINING_COMPLETE", summary, flush=True)
    return summary
