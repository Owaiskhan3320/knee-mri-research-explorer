"""Check a planned pooling comparison without importing models or reading data.

This is a design checker, not an experiment launcher. A valid design remains
disabled and deferred; no result, feasibility measurement or training is implied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INHERITED_SETTINGS = ("projection", "train_windows_per_slot", "validation_windows", "batch_size", "accumulation",
                      "epochs", "encoder_lr", "head_lr", "weight_decay", "min_delta", "warmup_steps", "intensity_jitter", "encoder_chunk")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, protocol=None):
    protocol_path = root / "docs/pooling_protocol.json"
    p = protocol if protocol is not None else json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg_path = root / "training/config/epoch8_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    provenance = json.loads((root / "training/provenance.json").read_text(encoding="utf-8"))
    anchors = p["source_anchors"]
    require(digest(cfg_path) == anchors["config_sha256"] == provenance["configuration"]["sha256"], "Source configuration changed.")
    for anchor, name in (("model_source_sha256", "cnn_convnext.py"), ("reader_source_sha256", "parent_cnn_training.py"),
                         ("historical_trainer_sha256", "convnext_train.py")):
        require(digest(root / "training/archive/convnext" / name) == anchors[anchor] == cfg["hashes"][name], "Archived source changed: " + name)
    require(anchors["historical_deployed_checkpoint_sha256"] == provenance["checkpoint"]["sha256"] and
            anchors["historical_checkpoint_is_training_initialization"] is False, "Historical trained weights cannot initialize this comparison.")
    require(p["status"] == "PLANNED_NOT_RUN" and p["results"] is None, "A planned protocol cannot claim results.")
    require(p["arms"] == [{"id":"attention_control", "pooling":"classwise_gated_attention"},
                          {"id":"mean_candidate", "pooling":"masked_global_mean"}],
            "Only pooling may differ between the two arms; per-arm overrides are forbidden.")
    common, data, init = p["common"], p["common"]["data"], p["common"]["initialization"]
    require(common["seed"] == cfg["seed"] and common["encoder"] == cfg["encoder"] and common["encoder_trainable"] is True and
            common["targets"] == cfg["targets"] and common["slots"] == cfg["slots"], "Shared model/seed/target contract differs.")
    require(init == {"source":"original_pretraining_initial_model.pt", "sha256":cfg["hashes"]["initial_model.pt"],
                     "shared_state_exact":True, "shared_prefixes":["encoder.","project.","slot."],
                     "shared_names":["finding_weight","finding_bias","mean","std"],
                     "attention_only_prefixes":["av.","au.","attention."], "reset_rng_after_model_construction":True},
            "Both arms need identical original shared initialization and reset training RNGs.")
    require(data["fingerprint"] == provenance["checkpoint"]["data_fingerprint"] and
            data["study_labels_splits_sha256"] == cfg["hashes"]["study_labels_splits.csv"] and
            data["pixel_index_sha256"] == cfg["hashes"]["pixel_index.csv"] and data["pixel_key"] == cfg["pixel_key"] and
            data["pixel_partition_manifest_sha256"] == [v["manifest_sha256"] for v in cfg["pixel_partitions"]] and
            data["split_sizes"] == cfg["expected_counts"] and data["label_source"] == cfg["label_source"], "Fixed data identities differ.")
    require(data["unknown_labels_masked"] is True and data["label_refinement"] is False and data["new_split"] is False and
            (data["size"], data["adjacent_slices_per_window"], data["crop_mm"]) == (224,3,140.0) and
            data["slot_embedding_preserved"] is True and data["normalization"] == "original uint8 /255 and one ImageNet normalization", "Labels or input preprocessing changed.")
    require(common["training_settings"] == {k:cfg["settings"][k] for k in INHERITED_SETTINGS}, "Inherited training settings differ.")
    require(common["optimizer"] == {"name":"AdamW", "betas":[.9,.999], "eps":1e-8,
                                    "gradient_clip_norm":5.0, "fresh_state_per_arm":True}, "Shared optimizer differs.")
    settings = common["training_settings"]
    batches = (data["split_sizes"]["fit"] + settings["batch_size"] - 1) // settings["batch_size"]
    groups = (batches + settings["accumulation"] - 1) // settings["accumulation"]
    require(common["schedule"] == {"kind":"linear_warmup_then_cosine", "floor_multiplier":.1,
                                   "planned_optimizer_updates":groups*settings["epochs"]}, "Schedule or matched exposure differs.")
    require(common["sampling"] == {"study_order":"original seed + zero-based epoch permutation", "window_sampling":"unchanged PixelBags epoch/UID rule",
                                   "augmentations":"unchanged intensity gain jitter", "record_matching_exogenous_hashes":True}, "Sampling/RNG contract differs.")
    require(common["early_stopping"]["enabled"] is False and
            common["numeric_failure_policy"] == "Abort the pair on nonfinite loss or gradients; no skipped CPU updates or automatic retry.", "Matched exposure/failure policy differs.")
    selection = common["checkpoint_selection"]
    require(selection == {"metric":cfg["selection"], "min_delta":cfg["settings"]["min_delta"], "complete_validation_only":True,
                          "independent_per_arm":True, "expert_used":False, "leaderboard_used":False}, "Checkpoint selection must use complete weak-validation BCE only.")
    evaluation = p["evaluation"]
    require(evaluation["primary_endpoint"] == "weak_validation_macro_auc_at_bce_selected_checkpoint" and
            evaluation["historical_0_833_is_control_result"] is False and evaluation["cutoff_fitted"] is False and
            evaluation["diagnostic_cutoff"] == .5 and evaluation["paired_study_sets_required"] is True and
            evaluation["complete_eight_epochs_per_arm_required"] is True, "Evaluation cannot reuse historical control scores or unequal exposure.")
    require(evaluation["candidate_screen"]["minimum_macro_auc_gain"] == .005 and
            evaluation["candidate_screen"]["maximum_soft_bce_increase"] == 0.0, "Predeclared pilot screen differs.")
    execution = p["execution"]
    require(execution["run_enabled"] is False and execution["trainer_implemented"] is False and
            execution["allowed_device"] == "cpu" and execution["precision"] == "float32" and
            execution["autocast"] is False and execution["grad_scaler"] is False and execution["gpu_runner_hours"] == 0 and
            execution["paid_services_allowed"] is False, "Execution must remain disabled, CPU-only and free of paid services.")
    require(execution["cpu_threads"] == 2 and execution["loader_workers"] == 0 and execution["seed_pairs"] == 1 and
            execution["sequential_arms"] is True and execution["automatic_retry"] is False and execution["automatic_resume"] is False and
            0 < execution["maximum_total_wall_minutes"] <= 120 and 0 < execution["maximum_profile_wall_minutes"] <= 5 and
            0 < execution["maximum_process_rss_gib"] <= 4, "Proposed CPU resource ceiling differs.")
    feasibility = p["feasibility"]
    require(feasibility["state"] == "DEFERRED" and feasibility["full_pixel_cache_verified_locally"] is False and
            feasibility["paired_cpu_training_timing_measured"] is False and feasibility["projected_total_wall_minutes"] is None and
            feasibility["projection_safety_factor"] == 1.5, "Unmeasured feasibility cannot be reported as ready.")
    profile = feasibility["profiling_protocol"]
    require(profile["training_groups_per_arm"] == {"warmup":3, "timed":5, "full_accumulation_group_required":True} and
            profile["validation_batches_per_arm"] == {"warmup":1, "timed":3, "windows_per_study":40} and
            profile["saving_reserve_minutes"] == 5, "Representative paired profiling requirements differ.")
    # This is a source-shape calculation, not a constructed or trained candidate.
    projection, targets = settings["projection"], len(common["targets"])
    attention_parameters = 2*(projection*128+128) + 128*targets+targets
    return {"status":"POOLING_PROTOCOL_VALID", "recorded_date":p["recorded_date"], "scheduled_task_date":p["scheduled_task_date"],
            "execution_status":"DEFERRED", "training_started":False, "models_constructed":False,
            "private_data_or_weights_loaded":False, "gpu_used":False, "paid_services_used":False,
            "one_intended_arm_difference":"pooling", "epochs_per_arm":settings["epochs"], "optimizer_updates_per_arm":groups*settings["epochs"],
            "shared_initial_tensors_runtime_verified":False, "cpu_training_time_measured":False,
            "parameter_counts_derived_from_source_shapes":{"attention_control":cfg["parameter_count"],
                "mean_candidate":cfg["parameter_count"]-attention_parameters, "attention_only_difference":attention_parameters},
            "parameter_count_limitation":"Pooling-method comparison includes a capacity difference; it is not a capacity-matched study.",
            "results":None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        sys.exit(f"Protocol check failed: {exc}")
