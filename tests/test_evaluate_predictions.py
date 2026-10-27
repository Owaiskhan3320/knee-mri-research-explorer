"""Small artificial cases with known answers; no MRI data, weights or GPU."""
import csv
import hashlib
import json
import math
import random

import pytest

from scripts.evaluate_predictions import TARGETS, binary_auc, evaluate, read_csv, reconcile
from scripts import evaluate_predictions as evaluator


def fixture_tables():
    labels, predictions = [], []
    for index, (hard, soft, probability) in enumerate([(0, .2, .1), (1, .7, .8), (0, .1, .5), (1, .9, .5)]):
        label = {"StudyInstanceUID": f"synthetic-{index}", "split": "weak_validation"}
        pred = {"StudyInstanceUID": label["StudyInstanceUID"]}
        for target in TARGETS:
            label[target], label[target + "__binary"], pred[target] = soft, hard, probability
        labels.append(label)
        predictions.append(pred)
    return predictions, labels


def test_auc_ties_and_direction_have_independent_known_answers():
    assert binary_auc([0, 1, 0, 1], [.1, .8, .5, .5]) == .875
    assert binary_auc([0, 1], [.5, .5]) == .5
    assert binary_auc([0, 1], [.9, .1]) == 0
    assert binary_auc([0, 1], [.1, .9]) == 1


def test_auc_matches_exhaustive_pair_definition_for_random_ties():
    rng = random.Random(42)
    for _ in range(30):
        labels = [0] * 6 + [1] * 5
        scores = [rng.choice([.1, .2, .4, .8]) for _ in labels]
        positive = [p for y, p in zip(labels, scores) if y]
        negative = [p for y, p in zip(labels, scores) if not y]
        expected = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / 30
        assert binary_auc(labels, scores) == expected


def test_alignment_and_named_target_order_are_independent_of_row_column_order():
    predictions, labels = fixture_tables()
    before = evaluate(predictions, labels, "weak_validation")
    predictions = [dict(reversed(list(row.items()))) for row in reversed(predictions)]
    assert evaluate(predictions, list(reversed(labels)), "weak_validation") == before
    assert before["macro_auc"] == .875
    assert before["targets"] == list(TARGETS)


def test_unknowns_are_excluded_from_auc_and_soft_bce_not_replaced_with_negatives():
    predictions, labels = fixture_tables()
    labels[1]["ACL"] = labels[1]["ACL__binary"] = ""
    row = evaluate(predictions, labels, "weak_validation")["per_target"][0]
    assert (row["known"], row["unknown"], row["positives"], row["negatives"]) == (3, 1, 1, 2)
    assert row["auc"] == .75
    expected = sum(-y * math.log(p) - (1-y) * math.log(1-p) for y, p in [(.2, .1), (.1, .5), (.9, .5)]) / 3
    assert row["soft_bce"] == pytest.approx(expected)


def test_single_class_and_no_known_labels_are_explicit_and_not_macro_zeroes():
    predictions, labels = fixture_tables()
    for row in labels:
        row["ACL"], row["ACL__binary"] = .1, 0
        row["MCL"] = row["MCL__binary"] = "nan"
    result = evaluate(predictions, labels, "weak_validation")
    assert result["per_target"][0]["auc_status"] == "single_class"
    assert result["per_target"][1]["auc_status"] == "no_known_labels"
    assert result["macro_auc_defined_targets"] == 10
    assert result["macro_auc"] == .875


@pytest.mark.parametrize("problem", ["duplicate_prediction", "duplicate_label", "missing_prediction", "extra_prediction", "mask", "soft_binary", "group_leak"])
def test_bad_input_sets_masks_and_labels_fail(problem):
    predictions, labels = fixture_tables()
    if problem == "duplicate_prediction":
        predictions.append(predictions[0])
    elif problem == "duplicate_label":
        labels.append(labels[0])
    elif problem == "missing_prediction":
        predictions.pop()
    elif problem == "extra_prediction":
        predictions.append(dict(predictions[0], StudyInstanceUID="unexpected-synthetic"))
    elif problem == "mask":
        labels[0]["ACL"] = ""
    elif problem == "soft_binary":
        labels[0]["ACL__binary"] = .4
    else:
        for index, row in enumerate(labels):
            row["verified_group"] = str(index)
        labels.append(dict(labels[0], StudyInstanceUID="synthetic-fit", split="fit"))
    with pytest.raises(ValueError):
        evaluate(predictions, labels, "weak_validation")


@pytest.mark.parametrize("probability", ["nan", "inf", -.1, 1.1, "bad", ""])
def test_invalid_predictions_are_rejected_even_when_label_unknown(probability):
    predictions, labels = fixture_tables()
    predictions[0]["ACL"] = probability
    labels[0]["ACL"] = labels[0]["ACL__binary"] = ""
    with pytest.raises(ValueError):
        evaluate(predictions, labels, "weak_validation")


def test_csv_duplicate_headers_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text("StudyInstanceUID,ACL,ACL\nsynthetic,.1,.9\n")
    with pytest.raises(ValueError, match="duplicate CSV columns"):
        read_csv(path)


def test_recorded_comparison_detects_a_wrong_metric():
    predictions, labels = fixture_tables()
    report = evaluate(predictions, labels, "weak_validation")
    saved = [{"target": t, "count": "4", "positives": "2", "auc": ".875"} for t in TARGETS]
    summary = {"complete": True, "candidate_ready": True, "best_epoch": 8, "weak_macro_auc": .875,
               "weak_validation_loss": report["masked_soft_bce_from_probabilities"]}
    assert reconcile(report, saved, summary)["status"] == "RECONCILED"
    saved[0]["auc"] = ".9"
    with pytest.raises(ValueError, match="per-finding AUC disagrees"):
        reconcile(report, saved, summary)


def test_cli_reconciles_complete_synthetic_split_and_writes_only_aggregates(tmp_path, monkeypatch):
    """Exercise CSV parsing, identity checks, reconciliation and JSON output together."""
    (tmp_path / "training/config").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    predictions, labels = fixture_tables()
    predictions = [dict(predictions[i % 4], StudyInstanceUID=f"synthetic-{i}") for i in range(637)]
    labels = [dict(labels[i % 4], StudyInstanceUID=f"synthetic-{i}") for i in range(637)]
    def save_csv(name, rows):
        path = private / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path
    pred_path = save_csv("val_predictions.csv", predictions)
    label_path = save_csv("study_labels_splits.csv", labels)
    expected = evaluate(predictions, labels, "weak_validation")
    auc_path = save_csv("val_auc.csv", [{"target": r["target"], "count": r["known"],
                                       "positives": r["positives"], "auc": r["auc"]} for r in expected["per_target"]])
    summary_path = private / "run_summary.json"
    summary_path.write_text(json.dumps({"complete": True, "candidate_ready": True, "best_epoch": 8,
                                      "weak_macro_auc": expected["macro_auc"], "data_fingerprint": "synthetic",
                                      "weak_validation_loss": expected["masked_soft_bce_from_probabilities"]}))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "training/config/epoch8_config.json").write_text(json.dumps({"targets": list(TARGETS),
                 "hashes": {"study_labels_splits.csv": digest(label_path)}}))
    (tmp_path / "training/provenance.json").write_text(json.dumps({"checkpoint": {
                 "data_fingerprint": "synthetic", "sha256": "synthetic-checkpoint"}}))
    (tmp_path / "docs/epoch8_evaluation_provenance.json").write_text(json.dumps({"recovered_files": {
                 p.name: {"sha256": digest(p)} for p in (pred_path, auc_path, summary_path)}}))
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    output = tmp_path / "aggregate.json"
    args = ["--predictions", str(pred_path), "--labels", str(label_path), "--split", "weak_validation",
            "--recorded-auc", str(auc_path), "--recorded-summary", str(summary_path), "--output", str(output)]
    evaluator.main(args)
    first = output.read_bytes()
    evaluator.main(args)
    assert output.read_bytes() == first
    result = json.loads(first)
    assert result["reconciliation"]["status"] == "RECONCILED"
    assert result["studies"] == 637
    assert "synthetic-0" not in first.decode() and str(private) not in first.decode()
    pred_path.write_bytes(pred_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Historical output hash differs"):
        evaluator.main(args)
    assert output.read_bytes() == first
