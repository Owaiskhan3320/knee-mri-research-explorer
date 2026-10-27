import json
from pathlib import Path
import shutil

import pytest

from scripts.verify_pooling_protocol import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def design_root(tmp_path):
    for relative in ("docs/pooling_protocol.json", "training/config/epoch8_config.json", "training/provenance.json",
                     "training/archive/convnext/cnn_convnext.py", "training/archive/convnext/parent_cnn_training.py",
                     "training/archive/convnext/convnext_train.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    return tmp_path


def protocol(root):
    return json.loads((root / "docs/pooling_protocol.json").read_text())


def test_design_verifies_without_data_weights_or_model_imports(design_root):
    assert not (design_root / "data").exists() and not (design_root / "models").exists()
    report = verify(design_root)
    assert report["status"] == "POOLING_PROTOCOL_VALID"
    assert report["execution_status"] == "DEFERRED"
    assert report["results"] is None
    assert report["optimizer_updates_per_arm"] == 3656


def test_per_arm_training_override_breaks_the_single_factor_comparison(design_root):
    p = protocol(design_root)
    p["arms"][1]["encoder_lr"] = 1e-5
    with pytest.raises(ValueError, match="Only pooling may differ"):
        verify(design_root, p)


def test_trained_checkpoint_cannot_replace_shared_pretraining_initialization(design_root):
    p = protocol(design_root)
    p["common"]["initialization"]["sha256"] = p["source_anchors"]["historical_deployed_checkpoint_sha256"]
    with pytest.raises(ValueError, match="original shared initialization"):
        verify(design_root, p)


@pytest.mark.parametrize("key,value", [("run_enabled",True), ("gpu_runner_hours",1), ("precision","float16")])
def test_design_cannot_enable_training_or_spend_gpu(design_root, key, value):
    p = protocol(design_root)
    p["execution"][key] = value
    with pytest.raises(ValueError, match="Execution must remain disabled"):
        verify(design_root, p)


def test_expert_selection_and_shortened_training_are_rejected(design_root):
    p = protocol(design_root)
    p["common"]["checkpoint_selection"]["expert_used"] = True
    with pytest.raises(ValueError, match="weak-validation BCE only"):
        verify(design_root, p)
    p = protocol(design_root)
    p["common"]["training_settings"]["epochs"] = 1
    with pytest.raises(ValueError, match="Inherited training settings differ"):
        verify(design_root, p)


def test_unrun_design_cannot_claim_a_score_or_measured_readiness(design_root):
    p = protocol(design_root)
    p["results"] = {"macro_auc":.96}
    with pytest.raises(ValueError, match="cannot claim results"):
        verify(design_root, p)
    p = protocol(design_root)
    p["feasibility"]["projected_total_wall_minutes"] = 60
    with pytest.raises(ValueError, match="Unmeasured feasibility"):
        verify(design_root, p)


def test_changed_archived_model_invalidates_source_binding(design_root):
    path = design_root / "training/archive/convnext/cnn_convnext.py"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="Archived source changed"):
        verify(design_root)
