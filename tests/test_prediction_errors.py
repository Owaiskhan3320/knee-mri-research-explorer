import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.analyze_prediction_errors import (analyze_split, build, coverage, error_profile,
                                               private_queue_path, write_queue)


def tables():
    labels, predictions = [], []
    # Include an unknown with a high score: it must not become a false positive.
    for i, (hard, score) in enumerate([(0, .1), (1, .8), (0, .5), (1, .5), (None, .99)]):
        labels.append({"StudyInstanceUID": f"synthetic-{i}", "split": "weak_validation",
                       "ACL": "" if hard is None else .2 if hard == 0 else .8,
                       "ACL__binary": "" if hard is None else hard})
        predictions.append({"StudyInstanceUID": f"synthetic-{i}", "ACL": score})
    return predictions, labels


def test_fixed_cutoff_and_ranking_pairs_have_known_answers():
    result = error_profile([0, 1, 0, 1], [.1, .8, .5, .5])
    assert [result[k] for k in ("true_positive", "false_negative", "false_positive", "true_negative")] == [2, 0, 1, 1]
    assert result["false_negative_fraction"] == 0
    assert result["false_positive_fraction"] == .5
    assert result["precision"] == pytest.approx(2/3)
    assert result["positive_negative_pairs"] == 4
    assert result["discordant_pairs"] == 0
    assert result["tied_pairs"] == 1
    assert result["ranking_error_fraction"] == .125
    assert result["auc"] == .875


def test_unknown_labels_are_neither_errors_nor_review_candidates():
    predictions, labels = tables()
    report, queue, _ = analyze_split(predictions, labels, "weak_validation", targets=("ACL",))
    assert report["per_target"][0]["known"] == 4
    assert [r["StudyInstanceUID"] for r in queue] == ["synthetic-2"]
    assert queue[0]["bucket_at_fixed_cutoff"] == "FP"
    assert "synthetic-" not in json.dumps(report)
    cov = coverage(labels, targets=("ACL",))["weak_validation"]["per_target"][0]
    assert (cov["known"], cov["unknown"], cov["positives"], cov["negatives"]) == (4, 1, 2, 2)
    assert cov["coverage_fraction"] == .8
    assert cov["positive_fraction_among_known"] == .5


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 0], [.1, .2]), ([1, 1], [.8, .9])])
def test_undefined_class_denominators_stay_null(labels, scores):
    result = error_profile(labels, scores)
    assert result["auc"] is None and result["ranking_error_fraction"] is None
    assert result["positive_negative_pairs"] == 0
    if 1 not in labels:
        assert result["recall"] is None and result["false_negative_fraction"] is None
    if 0 not in labels:
        assert result["specificity"] is None and result["false_positive_fraction"] is None


def test_perfect_and_inverted_ranking_are_distinct_from_cutoff_errors():
    perfect = error_profile([0, 0, 1, 1], [.1, .2, .3, .4])
    assert perfect["auc"] == 1 and perfect["ranking_error_fraction"] == 0
    assert perfect["false_negative"] == 2 and perfect["precision"] is None
    inverted = error_profile([0, 0, 1, 1], [.9, .8, .2, .1])
    assert inverted["ranking_error_fraction"] == 1 and inverted["auc"] == 0


def test_review_queue_is_bounded_sorted_and_independent_of_row_order(tmp_path):
    labels = [{"StudyInstanceUID": f"synthetic-{i:02d}", "split":"weak_validation", "ACL":.2, "ACL__binary":0} for i in range(8)]
    predictions = [{"StudyInstanceUID": r["StudyInstanceUID"], "ACL":.9} for r in labels]
    _, queue, _ = analyze_split(list(reversed(predictions)), labels, "weak_validation", targets=("ACL",))
    assert [r["StudyInstanceUID"] for r in queue] == [f"synthetic-{i:02d}" for i in range(5)]
    destination = private_queue_path(tmp_path / "data" / "queue.csv", root=tmp_path)
    write_queue(destination, queue)
    assert destination.read_text().count("synthetic-") == 5
    with pytest.raises(ValueError, match="must stay inside"):
        private_queue_path(tmp_path / "docs" / "queue.csv", root=tmp_path)


def test_coverage_rejects_soft_binary_mask_disagreement():
    _, labels = tables()
    labels[0]["ACL"] = ""
    with pytest.raises(ValueError, match="masks disagree"):
        coverage(labels, targets=("ACL",))


def test_changed_checkpoint_configuration_is_rejected(tmp_path):
    (tmp_path / "training/config").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    config = tmp_path / "training/config/epoch8_config.json"
    config.write_text('{}')
    (tmp_path / "training/provenance.json").write_text('{"configuration":{"sha256":"bad"}}')
    (tmp_path / "docs/epoch8_evaluation_provenance.json").write_text('{}')
    with pytest.raises(ValueError, match="Checkpoint configuration differs"):
        build(tmp_path, root=tmp_path)


def test_changed_historical_prediction_is_rejected_before_analysis(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "training/config").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    for relative in ("training/config/epoch8_config.json", "training/provenance.json"):
        shutil.copyfile(root / relative, tmp_path / relative)
    evidence = tmp_path / "private"
    evidence.mkdir()
    label_file = evidence / "study_labels_splits.csv"
    label_file.write_bytes(b"synthetic unused label bytes")
    (evidence / "val_predictions.csv").write_bytes(b"synthetic changed prediction bytes")
    provenance = {"recovered_files": {
        "study_labels_splits.csv":{"sha256":hashlib.sha256(label_file.read_bytes()).hexdigest()},
        "val_predictions.csv":{"sha256":"0"*64}}}
    (tmp_path / "docs/epoch8_evaluation_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Historical evidence hash differs: val_predictions.csv"):
        build(evidence, root=tmp_path)
