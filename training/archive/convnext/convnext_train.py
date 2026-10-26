"""One explicitly enabled, bounded ConvNeXt-Tiny training attempt.

Importing does not start work. The first 25 successful optimizer updates are
retained training: five warmup timings are excluded before throughput is gated.
This runner never retries, resumes, extends its budget, or submits predictions.
"""
import hashlib
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cnn_convnext as runtime


WARMUP_UPDATES = 5
BENCHMARK_UPDATES = 25
PROBE_WARMUP_BATCHES = 2
PROBE_TIMED_BATCHES = 8
SAFETY_FACTOR = 1.25
SAVE_RESERVE_SECONDS = 60.0


def timing_median(update_seconds, window=10):
    """Use successful updates after five warmup updates; never warmup samples."""
    usable = list(update_seconds)[WARMUP_UPDATES:]
    if not usable:
        return None
    return float(statistics.median(usable[-window:]))


def budget_projection(update_seconds, remaining_steps, validation_seconds,
                      gold_studies, validation_studies):
    per_update = timing_median(update_seconds)
    assert per_update is not None and per_update > 0
    assert remaining_steps >= 0 and validation_seconds > 0
    assert validation_studies > 0 and gold_studies >= 0
    gold_seconds = validation_seconds * gold_studies / validation_studies
    return dict(train_seconds=per_update * remaining_steps,
                validation_seconds=float(validation_seconds),
                gold_seconds=float(gold_seconds),
                save_reserve_seconds=SAVE_RESERVE_SECONDS,
                safety_factor=SAFETY_FACTOR,
                required_seconds=SAFETY_FACTOR *
                (per_update * remaining_steps + validation_seconds + gold_seconds)
                + SAVE_RESERVE_SECONDS)


def cosine_multiplier(step, planned_steps, warmup_steps=40):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    position = min(1.0, (step - warmup_steps) / max(1, planned_steps - warmup_steps))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * position))


def _settings(cfg):
    settings = dict(runtime.base.DEFAULTS, **cfg.get("settings", {}))
    settings["seed"] = int(cfg["seed"])
    assert settings["batch_size"] == 2 and settings["accumulation"] == 4
    assert settings["encoder_chunk"] == 32
    assert settings["train_windows_per_slot"] == 2
    assert settings["validation_windows"] == 40
    assert 1 <= settings["epochs"] <= 8 and 0 <= settings["workers"] <= 4
    assert settings["runtime_benchmark_required"] is True
    assert settings["execution_settings_require_gpu_profile"] is False
    assert settings["cudnn_benchmark"] is False
    assert settings["benchmark_warmup_optimizer_updates"] == WARMUP_UPDATES
    assert settings["benchmark_steady_optimizer_updates"] == BENCHMARK_UPDATES - WARMUP_UPDATES
    assert settings["validation_probe_warmup_batches"] == PROBE_WARMUP_BATCHES
    assert settings["validation_probe_timed_batches"] == PROBE_TIMED_BATCHES
    assert 0 <= settings["intensity_jitter"] <= .2
    return settings


def _loader(dataset, settings, batches=None):
    kwargs = dict(num_workers=settings["workers"], pin_memory=True,
                  persistent_workers=False)
    if settings["workers"]:
        kwargs["prefetch_factor"] = 2
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **kwargs)
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False, **kwargs)


def _device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def _predict(model, dataset, settings, device, deadline, reserve, probe=False):
    """Full deterministic bags; probe has two warm batches and eight timed ones.

    Batch wall time includes data read, host transfer submission and GPU work.
    Probe warmup/startup is explicitly recorded instead of extrapolated.
    """
    model.eval()
    rows, observations = [], []
    numerator_sum, known_count, complete = 0.0, 0, True
    started = time.monotonic()
    startup_started = time.monotonic()
    iterator = iter(_loader(dataset, settings))
    iterator_startup_seconds = time.monotonic() - startup_started
    with torch.inference_mode():
        for batch_index in range(math.ceil(len(dataset) / settings["batch_size"])):
            # Recent observations provide an additional per-batch deadline guard.
            next_seconds = max([item["wall_seconds"] for item in observations[-3:]] or [10.0])
            if time.monotonic() + reserve + SAFETY_FACTOR * next_seconds >= deadline:
                complete = False
                break
            batch_started = time.monotonic()
            read_started = time.monotonic()
            batch = next(iterator)
            read_seconds = time.monotonic() - read_started
            batch = _device_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch["images"], batch["valid"], batch["slots"],
                               settings["encoder_chunk"])
            numerator, known = runtime.loss_parts(logits, batch)
            probabilities = logits.float().sigmoid().cpu().numpy()
            assert np.isfinite(probabilities).all()
            rows.extend([[uid] + values.tolist()
                         for uid, values in zip(batch["uid"], probabilities)])
            numerator_sum += float(numerator)
            known_count += int(known)
            torch.cuda.synchronize(device)
            observations.append(dict(batch=batch_index + 1, studies=len(batch["uid"]),
                                     data_read_seconds=read_seconds,
                                     wall_seconds=time.monotonic() - batch_started,
                                     excluded_probe_warmup=probe and
                                     batch_index < PROBE_WARMUP_BATCHES))
            if probe and len(observations) >= PROBE_WARMUP_BATCHES + PROBE_TIMED_BATCHES:
                complete = len(rows) == len(dataset)
                break
    del iterator
    result = pd.DataFrame(rows, columns=["StudyInstanceUID"] + dataset.targets)
    profile = dict(complete=complete, observed_studies=len(result),
                   elapsed_seconds=time.monotonic() - started,
                   iterator_startup_seconds=iterator_startup_seconds,
                   data_read_seconds=sum(row["data_read_seconds"] for row in observations),
                   batches=observations)
    return result, numerator_sum / max(known_count, 1), complete, profile


def run_training(bundle_path, max_minutes=240,
                 output_path="/kaggle/working/knee_convnext_tiny"):
    """Run one budget-limited attempt. No retry/resume/extension is performed."""
    started = time.monotonic()
    assert 1 <= max_minutes <= 240
    deadline = started + max_minutes * 60
    root, cfg, frame, pixels = runtime.load_bundle(bundle_path, require_preflight=True)
    assert "convnext_train.py" in cfg["hashes"]
    assert runtime.sha256(__file__) == cfg["hashes"]["convnext_train.py"]
    assert max_minutes <= cfg.get("max_gpu_minutes", 240)
    settings = _settings(cfg)
    assert torch.cuda.is_available(), "Enable GPU only for this explicit bounded attempt"
    device = torch.device("cuda:0")
    out = Path(output_path)
    # An existing attempt must not be silently overwritten or resumed.
    assert not (out / "attempt_started.json").exists(), "Training attempt already exists"
    out.mkdir(parents=True, exist_ok=True)
    runtime.write_json(dict(started=True, automatic_retry=False, automatic_resume=False,
                            max_gpu_minutes=max_minutes, data_fingerprint=runtime.fingerprint(cfg)),
                       out / "attempt_started.json")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    model = runtime.model_from_bundle(root, require_preflight=True).to(
        device, memory_format=torch.channels_last)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=list(model.encoder.parameters()), lr=settings["encoder_lr"]),
        dict(params=[parameter for parameter in model.parameters()
                     if id(parameter) not in encoder_ids], lr=settings["head_lr"])],
        weight_decay=settings["weight_decay"])
    train_frame = frame.loc[frame.split.eq("fit")].reset_index(drop=True)
    val_frame = frame.loc[frame.split.eq("weak_validation")].reset_index(drop=True)
    gold_frame = frame.loc[frame.split.eq("gold")].reset_index(drop=True)
    assert (len(train_frame), len(val_frame), len(gold_frame)) == (3655, 637, 58)
    train = runtime.PixelBags(train_frame, pixels, cfg["targets"], cfg["pixel_key"],
                              training=True, seed=settings["seed"],
                              windows_per_slot=settings["train_windows_per_slot"])
    val = runtime.PixelBags(val_frame, pixels, cfg["targets"], cfg["pixel_key"])
    gold = runtime.PixelBags(gold_frame, pixels, cfg["targets"], cfg["pixel_key"])
    epoch_batches = math.ceil(len(train) / settings["batch_size"])
    epoch_steps = math.ceil(epoch_batches / settings["accumulation"])
    planned_steps = settings["epochs"] * epoch_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_multiplier(step, planned_steps, settings["warmup_steps"]))
    scaler = torch.amp.GradScaler("cuda")
    state = dict(epoch=0, batch_cursor=0, order=None, phase="train", optimizer_steps=0,
                 completed_epochs=0, best_loss=float("inf"), best_epoch=None,
                 bad_epochs=0, history=[], best_model=None, best_predictions=None,
                 train_loss_sum=0.0, train_known=0, skipped_updates=0,
                 consecutive_overflows=0, reason="running", successful_update_seconds=[],
                 group_observations=[], checkpoint_seconds=0.0,
                 validation_estimate_seconds=None, benchmark=None)

    def checkpoint(path):
        checkpoint_started = time.monotonic()
        runtime.atomic_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                                  progress=state, rng=_rng_state(),
                                  data_fingerprint=runtime.fingerprint(cfg), config=cfg,
                                  gpu_name=torch.cuda.get_device_name(0),
                                  resumable_at_optimizer_boundary=True,
                                  automatic_resume_permitted=False), path)
        state["checkpoint_seconds"] += time.monotonic() - checkpoint_started

    checkpoint(out / "last.pt")
    benchmark_viable, budget_stop, early_stop = None, False, False
    group_count = 0
    try:
        while state["epoch"] < settings["epochs"] and not early_stop:
            epoch = state["epoch"]
            epoch_started = time.monotonic()
            if state["order"] is None:
                state["order"] = np.random.default_rng(settings["seed"] + epoch).permutation(
                    len(train)).tolist()
            batches = [[(epoch, index) for index in
                        state["order"][position:position + settings["batch_size"]]]
                       for position in range(0, len(train), settings["batch_size"])]
            val_estimate = state["validation_estimate_seconds"]
            if state["batch_cursor"] == 0 and val_estimate is not None:
                projection = budget_projection(state["successful_update_seconds"], epoch_steps,
                                               val_estimate, len(gold), len(val))
                if time.monotonic() + projection["required_seconds"] >= deadline:
                    print("Next complete epoch plus validation/gold/save cannot fit; stopping.", flush=True)
                    budget_stop = True
                    break
            model.train()
            assert all(parameter.requires_grad for parameter in model.encoder.parameters())
            loader_started = time.monotonic()
            iterator = iter(_loader(train, settings, batches[state["batch_cursor"]:]))
            iterator_startup_seconds = time.monotonic() - loader_started
            print(dict(epoch=epoch + 1, iterator_startup_seconds=iterator_startup_seconds,
                       completed_epochs=state["completed_epochs"],
                       remaining_minutes=(deadline - time.monotonic()) / 60), flush=True)
            while state["batch_cursor"] < len(batches):
                val_estimate = state["validation_estimate_seconds"]
                recent_update = timing_median(state["successful_update_seconds"]) or 10.0
                validation_reserve = SAFETY_FACTOR * (val_estimate or 120.0)
                gold_reserve = validation_reserve * len(gold) / len(val)
                if time.monotonic() + SAVE_RESERVE_SECONDS + validation_reserve + \
                        gold_reserve + 2 * recent_update >= deadline:
                    budget_stop = True
                    break
                group_started = time.monotonic()
                read_started = time.monotonic()
                remaining = min(settings["accumulation"],
                                len(batches) - state["batch_cursor"])
                group = [next(iterator) for _ in range(remaining)]
                read_seconds = time.monotonic() - read_started
                denominator = sum(int(batch["mask"].sum()) for batch in group)
                assert denominator > 0
                optimizer.zero_grad(set_to_none=True)
                events = []
                host_compute_started = time.monotonic()
                for batch in group:
                    transfer_start, transfer_end, compute_end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    transfer_start.record()
                    batch = _device_batch(batch, device)
                    transfer_end.record()
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(runtime.augment(batch["images"], settings["intensity_jitter"]),
                                       batch["valid"], batch["slots"],
                                       settings["encoder_chunk"])
                    numerator, known = runtime.loss_parts(logits, batch)
                    assert bool(torch.isfinite(numerator)), "Nonfinite training loss"
                    scaler.scale(numerator / denominator).backward()
                    compute_end.record()
                    events.append((transfer_start, transfer_end, compute_end))
                    state["train_loss_sum"] += float(numerator.detach())
                    state["train_known"] += int(known)
                optimizer_start, optimizer_end = [torch.cuda.Event(enable_timing=True)
                                                  for _ in range(2)]
                optimizer_start.record()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_end.record()
                state["batch_cursor"] += remaining
                successful = scaler.get_scale() >= previous_scale
                if successful:
                    scheduler.step()
                    state["optimizer_steps"] += 1
                    state["consecutive_overflows"] = 0
                else:
                    state["skipped_updates"] += 1
                    state["consecutive_overflows"] += 1
                torch.cuda.synchronize(device)
                wall_seconds = time.monotonic() - group_started
                host_compute_seconds = time.monotonic() - host_compute_started
                group_count += 1
                observation = dict(group=group_count, epoch=epoch + 1,
                                   optimizer_updates=state["optimizer_steps"],
                                   successful_update=successful,
                                   studies=sum(len(batch["uid"]) for batch in group),
                                   data_read_seconds=read_seconds,
                                   transfer_cuda_seconds=sum(a.elapsed_time(b) for a, b, _ in events) / 1000,
                                   forward_backward_cuda_seconds=sum(b.elapsed_time(c) for _, b, c in events) / 1000,
                                   optimizer_cuda_seconds=optimizer_start.elapsed_time(optimizer_end) / 1000,
                                   host_compute_wall_seconds=host_compute_seconds,
                                   wall_seconds=wall_seconds,
                                   gradient_norm=float(gradient_norm),
                                   peak_memory_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
                                   elapsed_minutes=(time.monotonic() - started) / 60)
                state["group_observations"].append(observation)
                if successful:
                    state["successful_update_seconds"].append(wall_seconds)
                if group_count <= 30 or group_count % 50 == 0 or not successful:
                    print("PERFORMANCE", observation, flush=True)
                if not successful and state["consecutive_overflows"] >= 8:
                    checkpoint(out / "last.pt")
                    raise FloatingPointError("Eight consecutive AMP overflows; stopped without retry")
                if state["optimizer_steps"] and successful and state["optimizer_steps"] % 50 == 0:
                    checkpoint(out / "last.pt")
                if state["benchmark"] is None and state["optimizer_steps"] >= BENCHMARK_UPDATES:
                    probe_started = time.monotonic()
                    _, _, _, probe = _predict(model, val, settings, device, deadline,
                                               SAVE_RESERVE_SECONDS + gold_reserve, probe=True)
                    observations = probe["batches"]
                    timed = observations[PROBE_WARMUP_BATCHES:]
                    if len(timed) < PROBE_TIMED_BATCHES:
                        benchmark_viable, budget_stop = False, True
                        state["benchmark"] = dict(complete=False, probe=probe,
                                                   reason="Insufficient budget for full throughput probe")
                        runtime.write_json(state["benchmark"], out / "benchmark.json")
                        break
                    validation_per_batch = statistics.median(row["wall_seconds"] for row in timed)
                    # Account for initial loader/cache warmup once; never multiply
                    # these cold-start batches by the full validation batch count.
                    warmup_excess = max(0.0, sum(row["wall_seconds"] for row in
                                               observations[:PROBE_WARMUP_BATCHES]) -
                                        PROBE_WARMUP_BATCHES * validation_per_batch)
                    validation_startup = probe["iterator_startup_seconds"] + warmup_excess
                    val_estimate = validation_per_batch * math.ceil(len(val) / settings["batch_size"]) + validation_startup
                    state["validation_estimate_seconds"] = float(val_estimate)
                    remaining_steps = math.ceil((len(batches) - state["batch_cursor"]) /
                                                settings["accumulation"])
                    finish = budget_projection(state["successful_update_seconds"], remaining_steps,
                                               val_estimate, len(gold), len(val))
                    whole_epoch = budget_projection(state["successful_update_seconds"], epoch_steps,
                                                    val_estimate, len(gold), len(val))
                    benchmark_viable = time.monotonic() + finish["required_seconds"] < deadline
                    benchmark = dict(complete=True, gpu_used=True,
                                     encoder=cfg["encoder"], data_fingerprint=runtime.fingerprint(cfg),
                                     settings=settings, cudnn_benchmark=False,
                                     successful_updates=state["optimizer_steps"],
                                     excluded_warmup_updates=WARMUP_UPDATES,
                                     steady_update_observations=state["successful_update_seconds"][WARMUP_UPDATES:],
                                     median_last10_update_seconds=timing_median(state["successful_update_seconds"]),
                                     validation_warmup_batches=PROBE_WARMUP_BATCHES,
                                     validation_timed_batches=PROBE_TIMED_BATCHES,
                                     validation_probe=probe,
                                     validation_probe_total_seconds=time.monotonic() - probe_started,
                                     validation_probe_read_seconds=probe["data_read_seconds"],
                                     validation_startup_seconds_counted_once=validation_startup,
                                     validation_estimate_method="median eight warmed batches times full batch count plus one startup cost",
                                     validation_seconds_estimate=float(val_estimate),
                                     whole_epoch_projection=whole_epoch,
                                     current_epoch_remaining_projection=finish,
                                     first_complete_epoch_fits=benchmark_viable,
                                     remaining_budget_seconds=deadline - time.monotonic(),
                                     max_gpu_minutes=max_minutes)
                    state["benchmark"] = benchmark
                    runtime.write_json(benchmark, out / "benchmark.json")
                    print("CONVNEXT_RUNTIME_BENCHMARK", benchmark, flush=True)
                    model.train()
                    if not benchmark_viable:
                        print("One complete epoch plus full validation/gold/save cannot fit; no retry.", flush=True)
                        budget_stop = True
                        break
            del iterator
            if budget_stop:
                break
            state["phase"] = "validation"
            gold_reserve = SAFETY_FACTOR * (state["validation_estimate_seconds"] or 120.0) * len(gold) / len(val)
            predictions, val_loss, complete, validation_profile = _predict(
                model, val, settings, device, deadline, SAVE_RESERVE_SECONDS + gold_reserve)
            runtime.write_json(validation_profile, out / f"validation_profile_epoch{epoch + 1}.json")
            if not complete:
                predictions.to_csv(out / "val_predictions_partial.csv", index=False)
                budget_stop = True
                break
            assert len(predictions) == len(val) and predictions.StudyInstanceUID.is_unique
            state["validation_estimate_seconds"] = validation_profile["elapsed_seconds"]
            auc_table, auc = runtime.metrics(predictions, val_frame, cfg["targets"])
            assert np.isfinite(val_loss) and np.isfinite(auc)
            record = dict(epoch=epoch + 1,
                          training_loss=state["train_loss_sum"] / max(state["train_known"], 1),
                          weak_validation_loss=val_loss, weak_macro_auc=auc,
                          epoch_minutes=(time.monotonic() - epoch_started) / 60,
                          validation_minutes=validation_profile["elapsed_seconds"] / 60,
                          completed_optimizer_updates=state["optimizer_steps"])
            state["history"].append(record)
            print("EPOCH_COMPLETE", record, flush=True)
            improved = val_loss < state["best_loss"] - settings["min_delta"]
            if improved:
                state.update(best_loss=val_loss, best_epoch=epoch + 1, bad_epochs=0,
                             best_model={key: value.detach().cpu().clone()
                                         for key, value in model.state_dict().items()},
                             best_predictions=predictions.to_dict("list"))
            else:
                state["bad_epochs"] += 1
            state.update(epoch=epoch + 1, completed_epochs=epoch + 1, batch_cursor=0,
                         order=None, phase="train", train_loss_sum=0.0, train_known=0)
            pd.DataFrame(state["history"]).to_csv(out / "training_history.csv", index=False)
            if improved:
                checkpoint(out / "best.pt")
                auc_table.to_csv(out / "val_auc.csv", index=False)
            checkpoint(out / "last.pt")
            early_stop = state["bad_epochs"] >= settings["patience"]
    except Exception as exception:
        # The existing last.pt remains an optimizer-boundary checkpoint. A crash
        # inside an accumulation group must not overwrite it with partial state.
        runtime.write_json(dict(complete=False, failed=True, candidate_ready=state["best_model"] is not None,
                                exception_type=type(exception).__name__, message=str(exception),
                                automatic_retry=False, automatic_resume=False,
                                completed_epochs=state["completed_epochs"],
                                elapsed_gpu_job_minutes=(time.monotonic() - started) / 60,
                                data_fingerprint=runtime.fingerprint(cfg)), out / "failure_summary.json")
        raise

    state["reason"] = "budget_stopped" if budget_stop else "early_stopped" if early_stop else "epoch_limit"
    checkpoint(out / "last.pt")
    pd.DataFrame(state["group_observations"]).to_csv(out / "performance_groups.csv", index=False)
    summary = dict(complete=not budget_stop, candidate_ready=state["best_model"] is not None,
                   encoder=cfg["encoder"], benchmark_viable=benchmark_viable,
                   training_reason=state["reason"], planned_epochs=settings["epochs"],
                   completed_epochs=state["completed_epochs"], best_epoch=state["best_epoch"],
                   active_epoch=state["epoch"] + 1, next_batch=state["batch_cursor"],
                   optimizer_steps=state["optimizer_steps"], skipped_updates=state["skipped_updates"],
                   gpu_used=True, gpu_name=torch.cuda.get_device_name(0), max_gpu_minutes=max_minutes,
                   data_fingerprint=runtime.fingerprint(cfg), settings=settings,
                   cudnn_benchmark=False, checkpoint_seconds=state["checkpoint_seconds"],
                   expert_role="development diagnostic; never checkpoint or blend selection",
                   selection_metric="minimum complete 637-study masked soft-label validation BCE",
                   automatic_retry=False, automatic_resume=False,
                   submission_created=False, training_labels_changed=False)
    if state["best_model"] is not None:
        model.load_state_dict(state["best_model"], strict=True)
        selected = pd.DataFrame(state["best_predictions"])
        selected.to_csv(out / "val_predictions.csv", index=False)
        auc_table, auc = runtime.metrics(selected, val_frame, cfg["targets"])
        auc_table.to_csv(out / "val_auc.csv", index=False)
        summary.update(weak_validation_loss=state["best_loss"], weak_macro_auc=auc)
        runtime.atomic_torch(dict(model=state["best_model"], config=cfg,
                                  selected_epoch=state["best_epoch"], trained=True,
                                  targets=cfg["targets"], data_fingerprint=runtime.fingerprint(cfg)),
                             out / "deploy.pt")
        # The selected checkpoint and validation predictions are already frozen.
        gp, _, complete, gold_profile = _predict(model, gold, settings, device, deadline,
                                                  SAVE_RESERVE_SECONDS)
        summary["expert_evaluation_complete"] = complete
        runtime.write_json(gold_profile, out / "gold_prediction_profile.json")
        if complete:
            assert len(gp) == len(gold) and gp.StudyInstanceUID.is_unique
            gp.to_csv(out / "gold_predictions.csv", index=False)
            ga, gauc = runtime.metrics(gp, gold_frame, cfg["targets"])
            ga.to_csv(out / "gold_auc.csv", index=False)
            summary["expert_development_macro_auc"] = gauc
        else:
            gp.to_csv(out / "gold_predictions_partial.csv", index=False)
    summary["elapsed_gpu_job_minutes"] = (time.monotonic() - started) / 60
    runtime.write_json(summary, out / "run_summary.json")
    print("CONVNEXT_TRAINING_BUDGET_STOPPED" if budget_stop else "CONVNEXT_TRAINING_COMPLETE",
          summary, flush=True)
    return summary
