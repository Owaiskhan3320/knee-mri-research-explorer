import json

import pytest

from scripts.build_experiment_history import build, marker, performance_summary, selection_trace


def history_rows(losses, aucs=None):
    aucs = aucs or [.8] * len(losses)
    return [{"epoch": i, "training_loss": .4, "weak_validation_loss": loss,
             "weak_macro_auc": auc, "completed_optimizer_updates": i} for i, (loss, auc) in enumerate(zip(losses, aucs), 1)]


def test_selection_keeps_prior_winner_for_changes_below_min_delta():
    trace = selection_trace(history_rows([.6, .60001, .59995, .58], [.8, .9, .85, .84]), .0001)
    assert [r["selected_epoch_after_validation"] for r in trace] == [1, 1, 1, 4]
    assert [r["improved_by_selection_rule"] for r in trace] == [True, False, False, True]
    assert max(trace, key=lambda r:r["weak_macro_auc"])["epoch"] == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_invalid_loss_is_rejected(bad):
    with pytest.raises(ValueError, match="history metric"):
        selection_trace(history_rows([bad]), .0001)


def test_missing_epoch_cannot_be_filled_in():
    rows = history_rows([.6, .5])
    rows[1]["epoch"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        selection_trace(rows, .0001)


def test_skipped_update_is_distinct_from_a_failed_training_run():
    history = history_rows([.6])
    rows = [{"group": "1", "epoch": "1", "successful_update": "False", "optimizer_updates": "0", "gradient_norm": "inf"},
            {"group": "2", "epoch": "1", "successful_update": "True", "optimizer_updates": "1", "gradient_norm": "2.0"}]
    result = performance_summary(rows, history)
    assert result["successful_updates"] == 1
    assert result["skipped_updates"] == 1
    assert result["skipped_events"] == [{"group": 1, "epoch": 1, "gradient_norm_nonfinite": True}]
    rows[-1]["optimizer_updates"] = "2"
    with pytest.raises(ValueError, match="counter disagrees"):
        performance_summary(rows, history)


def test_saved_marker_parser_does_not_execute_notebook_source(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        marker("RESULT {'value': open(" + repr(str(sentinel)) + ", 'w')}", "RESULT ")
    assert not sentinel.exists()
    with pytest.raises(ValueError, match="exactly one"):
        marker("RESULT {}\nRESULT {}", "RESULT ")


def test_changed_private_evidence_is_rejected_before_report_generation(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "experiment_history_provenance.json").write_text(json.dumps({"private_files": {"history.csv": {"sha256": "0" * 64}}}))
    evidence = tmp_path / "private"
    evidence.mkdir()
    (evidence / "history.csv").write_text("synthetic changed evidence")
    with pytest.raises(ValueError, match="Historical evidence hash differs"):
        build(evidence, root=tmp_path)
