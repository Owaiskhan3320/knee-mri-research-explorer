"""Offline label coverage and disagreement analysis of saved epoch-8 outputs.

No model imports, inference, threshold fitting or training. Public output contains
only aggregates. An optional review queue contains private study identifiers.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_predictions import (TARGETS, UID, binary_auc, evaluate, index_rows,
                                          number, read_csv, reconcile, require, sha256)

THRESHOLD = 0.5
HIGH_SCORE = 0.9
LOW_SCORE = 0.1
QUEUE_PER_BUCKET = 5


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def coverage(labels, targets=TARGETS):
    index_rows(labels, "coverage labels")
    result = {}
    require(all(row.get("split") in {"fit", "weak_validation", "gold"} for row in labels), "Unexpected coverage split.")
    for split in ("fit", "weak_validation", "gold"):
        rows = [r for r in labels if r["split"] == split]
        per_target = []
        for target in targets:
            hard_labels = []
            for row in rows:
                soft = number(row[target], optional=True)
                hard = number(row[target + "__binary"], optional=True)
                require((soft is None) == (hard is None), "Coverage soft/binary masks disagree.")
                require(hard is None or hard in {0, 1}, "Coverage binary label must be zero or one.")
                if hard is not None:
                    hard_labels.append(int(hard))
            known, positives = len(hard_labels), sum(hard_labels)
            per_target.append({"target": target, "known": known, "unknown": len(rows)-known,
                               "positives": positives, "negatives": known-positives,
                               "coverage_fraction": ratio(known, len(rows)),
                               "positive_fraction_among_known": ratio(positives, known)})
        result[split] = {"studies": len(rows), "known_target_entries": sum(r["known"] for r in per_target),
                         "possible_target_entries": len(rows)*len(targets), "per_target": per_target}
    return result


def error_profile(labels, scores, threshold=THRESHOLD):
    require(len(labels) == len(scores) and 0 < threshold < 1, "Invalid error-profile inputs.")
    require(all(y in {0, 1} for y in labels), "Error profile requires explicit binary labels.")
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in scores), "Invalid prediction probability.")
    positives = [p for y, p in zip(labels, scores) if y == 1]
    negatives = [p for y, p in zip(labels, scores) if y == 0]
    tp, fp = sum(p >= threshold for p in positives), sum(p >= threshold for p in negatives)
    fn, tn = len(positives)-tp, len(negatives)-fp
    # These ranking counts use every known positive-negative pair, independent
    # of the diagnostic cutoff. Half a tied pair is an AUC ranking error.
    discordant = sum(p < n for p in positives for n in negatives)
    tied = sum(p == n for p in positives for n in negatives)
    pairs = len(positives)*len(negatives)
    ranking_error = ratio(discordant + .5*tied, pairs)
    auc = binary_auc(labels, scores)
    require(auc is None or abs(ranking_error - (1-auc)) <= 1e-12, "Ranking counts disagree with AUC.")
    return {"known": len(labels), "positives": len(positives), "negatives": len(negatives),
            "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
            "false_negative_fraction": ratio(fn, len(positives)), "false_positive_fraction": ratio(fp, len(negatives)),
            "precision": ratio(tp, tp+fp), "recall": ratio(tp, len(positives)),
            "specificity": ratio(tn, len(negatives)), "auc": auc,
            "positive_negative_pairs": pairs, "discordant_pairs": discordant, "tied_pairs": tied,
            "ranking_error_fraction": ranking_error,
            "positive_score_median": statistics.median(positives) if positives else None,
            "negative_score_median": statistics.median(negatives) if negatives else None,
            "negative_labels_with_score_at_least_0_9": sum(p >= HIGH_SCORE for p in negatives),
            "positive_labels_with_score_at_most_0_1": sum(p <= LOW_SCORE for p in positives)}


def analyze_split(predictions, labels, split, targets=TARGETS):
    # Reuse the strict completeness, ID alignment, group and unknown-mask checks.
    evaluated = evaluate(predictions, labels, split, targets)
    indexed = index_rows(labels, "analysis labels")
    profiles, queue = [], []
    for target, existing in zip(targets, evaluated["per_target"]):
        known, candidates = [], {"FP": [], "FN": []}
        for pred in predictions:
            row = indexed[pred[UID]]
            hard = number(row[target + "__binary"], optional=True)
            if hard is None:
                continue
            probability = number(pred[target])
            known.append((int(hard), probability))
            predicted = int(probability >= THRESHOLD)
            if predicted != hard:
                bucket = "FP" if predicted else "FN"
                # Priority means distance from the recorded binary label, not
                # calibrated confidence or a confirmed clinical error.
                candidates[bucket].append({UID: pred[UID], "split": split, "target": target,
                    "bucket_at_fixed_cutoff": bucket, "binary_label": int(hard),
                    "soft_label": number(row[target]), "score": probability,
                    "absolute_binary_label_disagreement": abs(hard-probability)})
        profile = error_profile([y for y, _ in known], [p for _, p in known])
        require(profile["known"] == existing["known"] and profile["auc"] == existing["auc"], "Evaluation/profile alignment differs.")
        profiles.append({"target": target, **profile})
        for bucket in ("FP", "FN"):
            queue.extend(sorted(candidates[bucket], key=lambda r:(-r["absolute_binary_label_disagreement"], r[UID]))[:QUEUE_PER_BUCKET])
    return {"studies": evaluated["studies"], "role": evaluated["role"], "per_target": profiles}, queue, evaluated


def build(evidence_dir, root=ROOT):
    config_path = root / "training/config/epoch8_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = json.loads((root / "training/provenance.json").read_text(encoding="utf-8"))
    recovered = json.loads((root / "docs/epoch8_evaluation_provenance.json").read_text(encoding="utf-8"))
    require(sha256(config_path) == training["configuration"]["sha256"] and config["targets"] == list(TARGETS), "Checkpoint configuration differs.")
    names = ("study_labels_splits.csv", "val_predictions.csv", "gold_predictions.csv", "val_auc.csv", "gold_auc.csv", "run_summary.json")
    for name in names:
        require(sha256(evidence_dir / name) == recovered["recovered_files"][name]["sha256"], "Historical evidence hash differs: " + name)
    require(sha256(evidence_dir / "study_labels_splits.csv") == config["hashes"]["study_labels_splits.csv"], "Checkpoint label-table hash differs.")
    _, labels = read_csv(evidence_dir / "study_labels_splits.csv")
    label_coverage = coverage(labels)
    require({s:r["studies"] for s, r in label_coverage.items()} == {"fit":3655, "weak_validation":637, "gold":58}, "Historical split counts differ.")
    summary = json.loads((evidence_dir / "run_summary.json").read_text(encoding="utf-8"))
    require(summary["data_fingerprint"] == training["checkpoint"]["data_fingerprint"], "Data fingerprint differs.")
    errors, queue = {}, []
    for split, prefix in (("weak_validation", "val"), ("gold", "gold")):
        header, predictions = read_csv(evidence_dir / (prefix + "_predictions.csv"))
        require(set(header) == {UID, *TARGETS}, "Unexpected prediction columns.")
        profile, review, evaluated = analyze_split(predictions, labels, split)
        _, recorded_auc = read_csv(evidence_dir / (prefix + "_auc.csv"))
        profile["reconciliation"] = reconcile(evaluated, recorded_auc, summary)
        errors[split] = profile
        queue.extend(review)
    report = {"schema_version":1, "recorded_date":"2026-10-27", "scheduled_task_date":"2026-10-29",
        "scope":"Existing epoch-8 predictions and labels; no model loading, inference or training.",
        "checkpoint_sha256":training["checkpoint"]["sha256"], "data_fingerprint":summary["data_fingerprint"],
        "protocol":{"diagnostic_cutoff":THRESHOLD, "positive_if":"score >= cutoff", "cutoff_fitted":False,
                    "high_score_negative_cutoff":HIGH_SCORE, "low_score_positive_cutoff":LOW_SCORE,
                    "ranking_error":"(discordant positive-negative pairs + half tied pairs) / all known positive-negative pairs",
                    "review_queue_policy":"Up to five largest absolute binary-label disagreements per finding/split/FP-or-FN bucket; ties sorted privately by study ID."},
        "label_coverage":label_coverage, "error_profiles":errors,
        "private_review_queue":{"rows":len(queue), "contains_study_identifiers":True, "published":False,
                                "review_status":"Queue created; scans and reports were not clinically adjudicated."},
        "input_sha256":{name:sha256(evidence_dir / name) for name in names},
        "limitations":["FP/FN are disagreements with recorded labels at a fixed illustrative cutoff, not confirmed clinical errors.",
            "Weak labels are report-derived; unknown findings stay masked rather than becoming negatives.",
            "Expert58 is small and repeatedly reused; these diagnostics do not estimate independent clinical performance.",
            "Different label sources and study populations prevent causal interpretation of differences between splits.",
            "No operating threshold, calibration, model or final competition selection was fitted or changed.",
            "No scanner/protocol/missing-sequence error association or radiologist review was established here."]}
    return report, queue


def private_queue_path(path, root=ROOT):
    resolved = path.resolve()
    require(any(resolved.is_relative_to((root / directory).resolve()) for directory in ("data", ".runtime")),
            "Review queue must stay inside the ignored project data/ or .runtime/ directory.")
    return resolved


def write_queue(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [UID, "split", "target", "bucket_at_fixed_cutoff", "binary_label", "soft_label", "score", "absolute_binary_label_disagreement"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    splits = ("fit", "weak_validation", "gold")
    matrix = [[report["label_coverage"][s]["per_target"][i]["coverage_fraction"] for s in splits] for i in range(len(TARGETS))]
    fig, ax = plt.subplots(figsize=(8.7, 8), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set(yticks=range(len(TARGETS)), yticklabels=TARGETS, xticks=range(3),
           xticklabels=["Fit (3655)", "Weak validation (637)", "Expert development (58)"],
           title="Known-label coverage by finding and split")
    ax.tick_params(length=0)
    for i in range(len(TARGETS)):
        for j, split in enumerate(splits):
            cell = report["label_coverage"][split]["per_target"][i]
            total = report["label_coverage"][split]["studies"]
            ax.text(j, i, f"{cell['known']}/{total}  ({cell['coverage_fraction']:.0%})",
                    ha="center", va="center", fontsize=8.5, color="white" if matrix[i][j]>.65 else "#162b3b")
    fig.colorbar(image, ax=ax, label="Fraction of studies with a known finding label", shrink=.75)
    fig.supxlabel("Unknown findings are masked; they are not negative labels.", fontsize=9)
    fig.savefig(directory / "label_coverage.png", dpi=160, metadata={"Software":"matplotlib " + matplotlib.__version__})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.8), sharey=True)
    fig.subplots_adjust(left=.12, right=.99, bottom=.17, top=.86, wspace=.08)
    for ax, split, title in zip(axes, ("weak_validation", "gold"), ("Weak validation: report-derived labels", "Expert development: 58 reused studies")):
        rows = report["error_profiles"][split]["per_target"]
        positions = list(range(len(TARGETS)))
        ax.barh([i-.18 for i in positions], [r["false_negative_fraction"] for r in rows], height=.34, color="#a95028", label="Below cutoff / positive labels (FN/P)")
        ax.barh([i+.18 for i in positions], [r["false_positive_fraction"] for r in rows], height=.34, color="#35628c", label="At/above cutoff / negative labels (FP/N)")
        ax.set(yticks=positions, yticklabels=TARGETS, xlim=(0,1), xlabel="Fraction of the corresponding known-label class", title=title)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5,.035), ncols=2, frameon=False, fontsize=9)
    fig.suptitle("Prediction-label disagreements at a fixed 0.5 cutoff", fontsize=14, y=.98)
    fig.text(.5,.015,"Diagnostic only: uncalibrated scores, different label sets, no clinical threshold or independent test claim.", ha="center", fontsize=9)
    fig.savefig(directory / "prediction_disagreements.png", dpi=160, metadata={"Software":"matplotlib " + matplotlib.__version__})
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args(argv)
    queue_path = private_queue_path(args.review_queue) if args.review_queue else None
    report, queue = build(args.evidence_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    if queue_path:
        write_queue(queue_path, queue)
    if args.plot_dir:
        plot(report, args.plot_dir)
    print(f"ERROR_ANALYSIS_RECONCILED: 12 findings; 4350 label rows; {len(queue)} private review entries; fixed cutoff 0.5.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        sys.exit(f"Error analysis failed: {exc}")
