"""Evaluate saved epoch-8 predictions offline; never imports a model or trains.

Only aggregate results and file hashes leave the private input tables. The CLI
requires the exact checkpoint-bound label/split table and complete study sets.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SOURCE_URL = "https://www.kaggle.com/code/owaiskhan15/knee-convnext-tiny-budget?scriptVersionId=360434584"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        require(bool(header) and len(header) == len(set(header)), "Empty or duplicate CSV columns.")
        rows = list(reader)
    require(bool(rows), "CSV has no data rows.")
    require(all(None not in row and None not in row.values() for row in rows), "Malformed CSV row width.")
    return header, rows


def index_rows(rows, role):
    indexed = {}
    for row in rows:
        uid = row.get(UID, "")
        require(isinstance(uid, str) and bool(uid.strip()), f"Missing study ID in {role}.")
        require(uid == uid.strip(), f"Padded study ID in {role}.")
        require(uid not in indexed, f"Duplicate study ID in {role}.")
        indexed[uid] = row
    return indexed


def number(value, *, optional=False):
    if optional and (value is None or str(value).strip().lower() in {"", "nan"}):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Non-numeric label or prediction.") from exc
    require(math.isfinite(result) and 0 <= result <= 1, "Nonfinite or out-of-range label/prediction.")
    return result


def binary_auc(labels, scores):
    """Mann-Whitney ROC AUC with half credit for tied scores; None if undefined."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    wins, negatives_before, start = 0.0, 0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        tied_positives = sum(y for _, y in ordered[start:end])
        tied_negatives = end - start - tied_positives
        wins += tied_positives * (negatives_before + 0.5 * tied_negatives)
        negatives_before += tied_negatives
        start = end
    return wins / (positives * negatives)


def soft_bce(label, probability):
    # Saved probabilities cannot recover the original logits at exact 0 or 1.
    # Reject those for BCE reconciliation rather than silently clipping scores.
    require(0 < probability < 1, "Endpoint probability: cannot reconcile original logit BCE.")
    return -label * math.log(probability) - (1 - label) * math.log1p(-probability)


def evaluate(predictions, labels, split, targets=TARGETS):
    require(split in {"weak_validation", "gold"}, "Unsupported evaluation split.")
    require(len(targets) == len(set(targets)), "Duplicate target names.")
    pred = index_rows(predictions, "predictions")
    truth = index_rows(labels, "labels")
    require(all(row.get("split") in {"fit", "weak_validation", "gold"} for row in labels), "Invalid label split.")
    selected = {uid: row for uid, row in truth.items() if row["split"] == split}
    require(bool(selected) and set(pred) == set(selected), "Prediction IDs must exactly match the complete requested split.")
    if any("verified_group" in row for row in labels):
        groups = {}
        for row in labels:
            group = row.get("verified_group", "")
            require(bool(group), "Missing verified group.")
            require(group not in groups or groups[group] == row["split"], "Verified group crosses splits.")
            groups[group] = row["split"]
    per_target, total_loss, total_known = [], [], 0
    for target in targets:
        binary, scores, losses = [], [], []
        for uid, row in selected.items():
            require(target in pred[uid] and target in row and target + "__binary" in row, "Missing named target column.")
            probability = number(pred[uid][target])  # Check even unknown-label predictions.
            soft = number(row[target], optional=True)
            hard = number(row[target + "__binary"], optional=True)
            require((soft is None) == (hard is None), "Soft and binary known-label masks disagree.")
            if hard is None:
                continue
            require(hard in {0, 1}, "AUC requires explicit binary labels; soft scores are not thresholded.")
            binary.append(int(hard))
            scores.append(probability)
            losses.append(soft_bce(soft, probability))
        known, positives = len(binary), sum(binary)
        auc = binary_auc(binary, scores)
        per_target.append({"target": target, "known": known, "unknown": len(selected) - known,
                           "positives": positives, "negatives": known - positives,
                           "auc": auc, "auc_status": "defined" if auc is not None else "no_known_labels" if not known else "single_class",
                           "soft_bce": math.fsum(losses) / known if known else None})
        total_loss.extend(losses)
        total_known += known
    defined = [row["auc"] for row in per_target if row["auc"] is not None]
    return {"split": split, "studies": len(selected), "targets": list(targets), "per_target": per_target,
            "macro_auc": math.fsum(defined) / len(defined) if defined else None,
            "macro_auc_defined_targets": len(defined), "macro_auc_total_targets": len(targets),
            "known_target_entries": total_known,
            "masked_soft_bce_from_probabilities": math.fsum(total_loss) / total_known if total_known else None,
            "role": "checkpoint-selection weak validation" if split == "weak_validation" else "reused expert development diagnostic; not independent test performance"}


def reconcile(report, recorded_auc, summary, *, auc_tolerance=1e-12, bce_tolerance=1e-6):
    """Fail before publishing if saved metrics or selected-run identity disagree."""
    require(len(recorded_auc) == len(TARGETS), "Incomplete recorded AUC table.")
    recorded = {row["target"]: row for row in recorded_auc}
    require(set(recorded) == set(TARGETS), "Duplicate, missing or unexpected recorded AUC targets.")
    deltas = []
    for row in report["per_target"]:
        old = recorded[row["target"]]
        require(int(old["count"]) == row["known"] and int(old["positives"]) == row["positives"], "Recorded label counts disagree.")
        old_auc = number(old["auc"], optional=True)
        require((old_auc is None) == (row["auc"] is None), "Recorded AUC definedness disagrees.")
        if old_auc is not None:
            deltas.append(abs(old_auc - row["auc"]))
    require(max(deltas, default=0) <= auc_tolerance, "Recomputed per-finding AUC disagrees with saved table.")
    require(summary.get("complete") is True and summary.get("candidate_ready") is True and summary.get("best_epoch") == 8,
            "Not the complete selected epoch-8 run.")
    key = "weak_macro_auc" if report["split"] == "weak_validation" else "expert_development_macro_auc"
    require(report["macro_auc"] is not None and abs(number(summary[key]) - report["macro_auc"]) <= auc_tolerance,
            "Recomputed macro AUC disagrees with run summary.")
    result = {"status": "RECONCILED", "maximum_per_target_auc_difference": max(deltas, default=0),
              "auc_absolute_tolerance": auc_tolerance, "bce_absolute_tolerance": bce_tolerance}
    if report["split"] == "weak_validation":
        delta = abs(number(summary["weak_validation_loss"]) - report["masked_soft_bce_from_probabilities"])
        require(delta <= bce_tolerance, "Probability-derived BCE disagrees with recorded logit BCE.")
        result["bce_difference_from_original_logit_loss"] = delta
    else:
        require(summary.get("expert_evaluation_complete") is True, "Saved expert evaluation is incomplete.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", choices=["weak_validation", "gold"], required=True)
    parser.add_argument("--recorded-auc", type=Path, required=True)
    parser.add_argument("--recorded-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads((ROOT / "training/config/epoch8_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "training/provenance.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "docs/epoch8_evaluation_provenance.json").read_text(encoding="utf-8"))
    require(config["targets"] == list(TARGETS), "Checkpoint target order differs from evaluator.")
    require(sha256(args.labels) == config["hashes"]["study_labels_splits.csv"], "Label/split table hash differs from checkpoint.")
    prefix = "val" if args.split == "weak_validation" else "gold"
    for path, name in [(args.predictions, prefix + "_predictions.csv"),
                       (args.recorded_auc, prefix + "_auc.csv"), (args.recorded_summary, "run_summary.json")]:
        require(sha256(path) == evidence["recovered_files"][name]["sha256"], "Historical output hash differs: " + name)
    header, predictions = read_csv(args.predictions)
    require(set(header) == {UID, *TARGETS}, "Predictions require only study ID and the 12 named finding columns.")
    _, labels = read_csv(args.labels)
    _, recorded = read_csv(args.recorded_auc)
    summary = json.loads(args.recorded_summary.read_text(encoding="utf-8"))
    require(summary["data_fingerprint"] == provenance["checkpoint"]["data_fingerprint"], "Saved summary data fingerprint differs.")
    report = evaluate(predictions, labels, args.split)
    require(report["studies"] == {"weak_validation": 637, "gold": 58}[args.split], "Unexpected historical split size.")
    report["reconciliation"] = reconcile(report, recorded, summary)
    report.update(schema_version=1, method="saved-probability evaluation; no inference or training",
                  source_notebook=SOURCE_URL, checkpoint_sha256=provenance["checkpoint"]["sha256"],
                  data_fingerprint=summary["data_fingerprint"], selected_epoch=8,
                  input_sha256={"predictions": sha256(args.predictions), "labels": sha256(args.labels),
                                "recorded_auc": sha256(args.recorded_auc), "recorded_summary": sha256(args.recorded_summary)})
    # No IDs, group assignments, individual labels/predictions or local paths.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"RECONCILED: {args.split}; {report['studies']} studies; macro AUC {report['macro_auc']:.12f}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        sys.exit(f"Evaluation failed: {exc}")
