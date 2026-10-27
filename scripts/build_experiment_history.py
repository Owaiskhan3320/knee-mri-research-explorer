"""Recover aggregate experiment history from hash-bound private logs, offline.

Never executes notebook cells or imports training code. The core report needs
only the standard library; --plot additionally uses optional matplotlib.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_output(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(output.get("text", [])) for cell in notebook["cells"]
                     for output in cell.get("outputs", []))


def marker(text, prefix, *, json_format=False):
    matches = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    require(len(matches) == 1, "Expected exactly one saved output marker: " + prefix)
    # literal_eval reads recorded dictionary data; it never executes source.
    result = json.loads(matches[0]) if json_format else ast.literal_eval(matches[0])
    require(isinstance(result, dict), "Saved marker is not a dictionary.")
    return result


def selection_trace(rows, min_delta):
    require(bool(rows) and math.isfinite(min_delta) and min_delta >= 0, "Invalid selection inputs.")
    selected, best, trace = None, math.inf, []
    for index, row in enumerate(rows, 1):
        require(int(row["epoch"]) == index, "Epochs must be complete, consecutive and start at one.")
        loss, auc, train = (float(row[k]) for k in ("weak_validation_loss", "weak_macro_auc", "training_loss"))
        require(all(math.isfinite(x) for x in (loss, auc, train)) and loss >= 0 and train >= 0 and 0 <= auc <= 1,
                "Nonfinite or out-of-range history metric.")
        improved = loss < best - min_delta
        if improved:
            selected, best = index, loss
        trace.append({**row, "improved_by_selection_rule": improved, "selected_epoch_after_validation": selected})
    return trace


def performance_summary(rows, history):
    require(bool(rows), "Missing optimizer observations.")
    updates, skipped, by_epoch = 0, [], {}
    for group, row in enumerate(rows, 1):
        require(int(row["group"]) == group and row["successful_update"] in {"True", "False"}, "Invalid optimizer observation order.")
        successful = row["successful_update"] == "True"
        updates += int(successful)
        require(int(row["optimizer_updates"]) == updates, "Successful-update counter disagrees with observations.")
        epoch = int(row["epoch"])
        require(1 <= epoch <= len(history), "Optimizer observation outside completed epochs.")
        require(math.isfinite(float(row["gradient_norm"])) or not successful, "Successful update has nonfinite gradients.")
        by_epoch[epoch] = updates
        if not successful:
            skipped.append({"group": group, "epoch": epoch, "gradient_norm_nonfinite": not math.isfinite(float(row["gradient_norm"]))})
    require(set(by_epoch) == set(range(1, len(history) + 1)), "Missing optimizer epoch observations.")
    require(all(by_epoch[int(row["epoch"])] == row["completed_optimizer_updates"] for row in history),
            "Epoch history and optimizer counters disagree.")
    return {"accumulation_groups": len(rows), "successful_updates": updates,
            "skipped_updates": len(skipped), "skipped_events": skipped}


def build(evidence_dir, root=ROOT):
    provenance = json.loads((root / "docs/experiment_history_provenance.json").read_text(encoding="utf-8"))
    for name, record in provenance["private_files"].items():
        path = evidence_dir / name
        require(path.is_file() and digest(path) == record["sha256"], "Historical evidence hash differs: " + name)
    evaluation_provenance = json.loads((root / "docs/epoch8_evaluation_provenance.json").read_text(encoding="utf-8"))
    for name in ("training_history.csv", "run_summary.json"):
        require(digest(evidence_dir / name) == evaluation_provenance["recovered_files"][name]["sha256"],
                "History differs from the independently recovered evaluation evidence.")
    config = json.loads((root / "training/config/epoch8_config.json").read_text(encoding="utf-8"))
    original_provenance = json.loads((root / "training/provenance.json").read_text(encoding="utf-8"))
    require(digest(root / "training/config/epoch8_config.json") == original_provenance["configuration"]["sha256"], "Checkpoint configuration changed.")
    with (evidence_dir / "training_history.csv").open(encoding="utf-8", newline="") as stream:
        history = [{k: int(v) if k in {"epoch", "completed_optimizer_updates"} else float(v) for k, v in row.items()}
                   for row in csv.DictReader(stream)]
    summary = json.loads((evidence_dir / "run_summary.json").read_text(encoding="utf-8"))
    settings = config["settings"]
    require(summary["data_fingerprint"] == original_provenance["checkpoint"]["data_fingerprint"], "Data fingerprint differs.")
    require(summary["complete"] is True and summary["training_reason"] == "epoch_limit" and
            len(history) == summary["completed_epochs"] == settings["epochs"] == 8, "Original training did not complete eight epochs.")
    trace = selection_trace(history, settings["min_delta"])
    selected = trace[-1]["selected_epoch_after_validation"]
    require(selected == summary["best_epoch"] == original_provenance["checkpoint"]["selected_epoch"] == 8, "Checkpoint-selection replay disagrees.")
    require(history[selected-1]["weak_validation_loss"] == summary["weak_validation_loss"] and
            history[selected-1]["weak_macro_auc"] == summary["weak_macro_auc"], "Selected history and summary metrics disagree.")
    with (evidence_dir / "performance_groups.csv").open(encoding="utf-8", newline="") as stream:
        optimizer = performance_summary(list(csv.DictReader(stream)), history)
    require(optimizer["successful_updates"] == summary["optimizer_steps"] and
            optimizer["skipped_updates"] == summary["skipped_updates"], "Optimizer summary disagrees.")
    benchmark = json.loads((evidence_dir / "benchmark.json").read_text(encoding="utf-8"))
    require(benchmark["data_fingerprint"] == summary["data_fingerprint"] and benchmark["complete"] is True and
            benchmark["first_complete_epoch_fits"] is True, "Runtime benchmark evidence differs.")

    efficient = marker(notebook_output(evidence_dir / "knee-efficientnet-b0-budget.ipynb"), "TRAINING_BUDGET_STOPPED ")
    require(efficient["completed_epochs"] == 0 and efficient["candidate_ready"] is False and efficient["benchmark_viable"] is False,
            "Stopped baseline evidence no longer matches the documented outcome.")
    refinement_text = notebook_output(evidence_dir / "knee-convnext-label-refinement.ipynb")
    refinement = marker(refinement_text, "RETRY_RESULT ", json_format=True)
    extra_epoch = marker(refinement_text, "EPOCH_COMPLETE ")
    require(refinement["completed_epochs"] == 1 and refinement["best_epoch"] == 0 and refinement["training_reason"] == "budget_stopped",
            "Refinement selection evidence differs.")
    require(refinement["weak_validation_loss"] == summary["weak_validation_loss"] and
            extra_epoch["weak_validation_loss"] > summary["weak_validation_loss"], "Refinement did not retain the original BCE winner.")
    expert = marker(notebook_output(evidence_dir / "epoch9_expert_v1.ipynb"), "TRIAL_EXPERT_RESULT ", json_format=True)
    trial = marker(notebook_output(evidence_dir / "knee-convnext-epoch9-cpu-evaluation.ipynb"), "TRIAL_SUBMISSION_READY ", json_format=True)
    require(expert["last_checkpoint_sha256"] == trial["last_checkpoint_sha256"] and
            expert["source_training_version"] == trial["source_training_version"] == 361212433 and
            expert["data_fingerprint"] == trial["data_fingerprint"] == refinement["data_fingerprint"] and
            expert["baseline_expert_development_auc"] == summary["expert_development_macro_auc"] and
            expert["trial_is_bce_winner"] is trial["trial_is_bce_winner"] is False, "Trial checkpoint/evaluation identities disagree.")
    scores = {}
    for line in (evidence_dir / "score_rows.txt").read_text(encoding="utf-8").splitlines():
        match = re.search(r"private score ([0-9.]+) public score ([0-9.]+)", line)
        require(match is not None, "Missing paired public/private score labels.")
        key = "epoch9" if "Knee ConvNeXt Epoch9 CPU Evaluation" in line else "epoch8" if "Knee ConvNeXt Verified Inference" in line else None
        require(key is not None and key not in scores, "Unexpected or duplicate submission row.")
        scores[key] = {"public": float(match[2]), "private": float(match[1])}
    require(set(scores) == {"epoch8", "epoch9"}, "Missing paired submission scores.")
    urls = provenance["source_urls"]
    return {"schema_version": 1, "recorded_date": provenance["recorded_date"], "scheduled_task": "28 October experiment-history task",
            "scope": "Historical artifacts only; no new training, inference, GPU use or payment.",
            "original_run": {"source_notebook": urls["epoch8"], "checkpoint_sha256": original_provenance["checkpoint"]["sha256"],
                "data_fingerprint": summary["data_fingerprint"], "epochs": trace, "optimizer": optimizer,
                "selection": {"rule": "validation_loss < best_loss - min_delta", "min_delta": settings["min_delta"],
                    "patience": settings["patience"], "selected_epoch": selected,
                    "highest_validation_auc_epoch": max(history, key=lambda r:r["weak_macro_auc"])["epoch"],
                    "training_end_reason": summary["training_reason"], "expert_used_for_checkpoint_selection": False},
                "historical_training_function_minutes": summary["elapsed_gpu_job_minutes"],
                "runtime_benchmark": {key: benchmark[key] for key in ("successful_updates", "excluded_warmup_updates", "median_last10_update_seconds", "first_complete_epoch_fits")},
                "historical_kaggle_displayed_runtime": "32m 11s; GPU T4 x2; trainer addresses cuda:0"},
            "experiments": [
                {"id": "efficientnet-budget-gate", "source_notebook": urls["efficientnet"],
                 "outcome": "budget_stopped_no_validated_candidate", "completed_epochs": efficient["completed_epochs"],
                 "candidate_ready": efficient["candidate_ready"], "historical_training_function_minutes": efficient["elapsed_gpu_job_minutes"],
                 "auc": None, "limitation": "No full validation/checkpoint candidate; throughput observations are not a controlled encoder comparison."},
                {"id": "convnext-epoch8", "source_notebook": urls["epoch8"], "outcome": "deployed_original_bce_winner",
                 "weak_validation_bce": summary["weak_validation_loss"], "weak_validation_auc": summary["weak_macro_auc"],
                 "expert_development_auc": summary["expert_development_macro_auc"], "competition_scores": scores["epoch8"]},
                {"id": "convnext-fixed-extra-epoch", "source_notebook": urls["refinement"],
                 "outcome": "one_extra_epoch_completed_budget_stopped_original_checkpoint_retained",
                 "completed_extra_epochs": 1, "optimizer_updates": extra_epoch["completed_optimizer_updates"],
                 "training_loss_changed_targets": extra_epoch["training_loss"],
                 "weak_validation_bce": extra_epoch["weak_validation_loss"], "weak_validation_auc": extra_epoch["weak_macro_auc"],
                 "selected_relative_epoch": refinement["best_epoch"], "selected_relative_epoch_zero_means": "original eight-epoch checkpoint",
                 "reported_selected_checkpoint_expert_auc": refinement["expert_development_macro_auc"],
                 "historical_training_function_minutes": refinement["elapsed_gpu_job_minutes"],
                 "confounds": ["training labels changed", "one additional epoch", "lower encoder and head learning rates", "new optimizer and warmup/cosine schedule"]},
                {"id": "convnext-epoch9-fixed-last-checkpoint-evaluation", "source_notebook": urls["epoch9_expert"],
                 "submission_notebook": urls["epoch9_submission"], "outcome": "fixed_last_checkpoint_trial_not_bce_winner",
                 "checkpoint_sha256": trial["last_checkpoint_sha256"], "expert_development_auc": expert["expert_development_auc"],
                 "expert_metric_evidence": "saved CPU evaluation log; not recalculated from trial predictions in this recovery",
                 "competition_scores": scores["epoch9"], "deployed": False}],
            "limitations": ["Expert58 was repeatedly reused; the refinement label source also used it in development.",
                "Training losses before and after the label-policy change optimize different targets and cannot establish improvement.",
                "Public/private scores are displayed to three decimals; these small differences are descriptive, not significance estimates.",
                "This is a partial artifact-backed history. Earlier ResNet18/DINO and refinement versions 1-4 are not fully reconstructed.",
                "No missing epoch curves, failed-run metrics, independent validation or controlled ablation results are invented."]}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["original_run"]["epochs"]
    epochs = [r["epoch"] for r in rows]
    selected = report["original_run"]["selection"]["selected_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    axes[0].plot(epochs, [r["training_loss"] for r in rows], "o-", color="#315b96", label="Training soft BCE")
    axes[0].plot(epochs, [r["weak_validation_loss"] for r in rows], "s-", color="#a54f24", label="Weak-validation soft BCE")
    axes[0].set(ylabel="Masked soft-label BCE", title="Original eight-epoch loss history")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(epochs, [r["weak_macro_auc"] for r in rows], "o-", color="#246b58")
    axes[1].set(ylabel="Weak-validation macro ROC AUC", title="AUC was monitored; BCE selected epoch 8")
    axes[1].annotate("Highest AUC: epoch 7", xy=(7, rows[6]["weak_macro_auc"]), xytext=(3.6, .805),
                     arrowprops={"arrowstyle": "->", "color": "#53606e"}, fontsize=9)
    for ax in axes:
        ax.axvline(selected, color="#657080", linestyle="--", linewidth=1, label="Selected epoch 8")
        ax.set(xticks=epochs, xlabel="Completed epoch", xlim=(.7, 8.35))
        ax.grid(alpha=.18)
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle("ConvNeXt-Tiny MIL — recovered saved logs", fontsize=14)
    fig.supxlabel("Validation guides model selection; this is not independent test performance.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, metadata={"Software": "matplotlib " + matplotlib.__version__})
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    report = build(args.evidence_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    if args.plot:
        plot(report, args.plot)
    print("HISTORY_RECONCILED: eight epochs; selected epoch 8; 3655 successful updates; one skipped update.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, SyntaxError) as exc:
        sys.exit(f"History recovery failed: {exc}")
