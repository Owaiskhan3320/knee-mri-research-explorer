"""Verify historical training source/configuration with the standard library only.

Never imports the archived modules, PyTorch, data or weights; never trains.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "e30535be63503f489d3e0961afa51cce83bfe7d9766faffbe5dfc2be454310dd"
FINGERPRINT = "f5116e8e4836c86bfd7f551ab8eaa182efb1bd12dc2742883c5259a1e0b122a3"
ORIGINAL_CELL_SHA = "847f988b911d8c16ea7d2bf2101885585b8e8990577ec7e8da142a2a2006cd5a"
SOURCE_HASHES = {
    "archive/convnext/cnn_convnext.py": "2b48dbe18a32e4a5e9c8c9adb83250e0f399c49343ff5cda6b0155bd855f29cd",
    "archive/convnext/parent_cnn_training.py": "fe38ece1a0a6e692d45f20e1f114cc876cbda16f97aec09c840e3adbc69a6ca2",
    "archive/convnext/convnext_train.py": "ac3516fee36c3a8b0b6191592e09844de52ab6803d41eda779ac4dbf912a6f3a",
    "archive/convnext/prepare_convnext.py": "9052cd372119aaad01cbb7e32a1d530a9f05d099ce6fc443be4d334f9101bb0b",
    "archive/parent_preparation/cnn_training.py": "fe38ece1a0a6e692d45f20e1f114cc876cbda16f97aec09c840e3adbc69a6ca2",
    "archive/parent_preparation/prepare.py": "a0f5f5b2095f839234109e84cb50774e81a4bd99f6a55d41b3ae609b129af14c",
    "archive/parent_preparation/label_audit.py": "c5bdf72e28c0d1e2b6b52b9f1ebcd58b0f7040e055192f3ea0f87b20d12c90e8",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def model_class(content):
    tree = ast.parse(content)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StudyMIL"]
    require(len(nodes) == 1, "Expected exactly one StudyMIL class.")
    return ast.dump(nodes[0], include_attributes=False)


def verify(root=ROOT):
    folder = root / "training"
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    require(set(provenance["files"]) == set(SOURCE_HASHES), "Unexpected archived source inventory.")
    for relative, expected in SOURCE_HASHES.items():
        path = folder / relative
        require(not path.is_symlink(), "Symlinks are not archived sources.")
        content = path.read_bytes()
        require(digest(content) == expected, "Archived source changed: " + relative)
        require(provenance["files"][relative]["sha256"] == expected, "Source provenance disagrees: " + relative)
        require(provenance["files"][relative]["bytes"] == len(content), "Source size disagrees: " + relative)
        ast.parse(content, filename=relative)

    config_bytes = (folder / "config" / "epoch8_config.json").read_bytes()
    require(digest(config_bytes) == provenance["configuration"]["sha256"], "Published configuration hash changed.")
    cfg = json.loads(config_bytes)
    clean = {k: v for k, v in cfg.items() if k != "hashes"}
    clean["hashes"] = {k: v for k, v in cfg["hashes"].items() if k != "preflight.json"}
    require(digest(json.dumps(clean, sort_keys=True).encode()) == FINGERPRINT,
            "Configuration does not match the historical data fingerprint.")
    for name in ("cnn_convnext.py", "parent_cnn_training.py", "convnext_train.py", "prepare_convnext.py"):
        require(cfg["hashes"][name] == SOURCE_HASHES["archive/convnext/" + name], "Config source binding disagrees.")
    require(cfg["seed"] == 42 and cfg["expected_counts"] == {"fit": 3655, "weak_validation": 637, "gold": 58},
            "Unexpected seed or split counts.")
    require(provenance["checkpoint"]["data_fingerprint"] == FINGERPRINT, "Checkpoint fingerprint disagrees.")
    require(provenance["checkpoint"]["sha256"] == CHECKPOINT_SHA and provenance["checkpoint"]["selected_epoch"] == 8,
            "Unexpected checkpoint identity or selected epoch.")
    serving = json.loads((root / "models" / "manifest.json").read_text())
    require(serving["checkpoint_sha256"] == CHECKPOINT_SHA, "Serving checkpoint identity disagrees.")
    original = (folder / "archive" / "convnext" / "cnn_convnext.py").read_bytes()
    require(serving["original_model_source_sha256"] == digest(original), "Serving source provenance disagrees.")
    require(model_class(original) == model_class((root / "app" / "model.py").read_bytes()),
            "Archived and serving model classes differ.")

    notebook_bytes = (folder / "notebooks" / "convnext_tiny_budget.ipynb").read_bytes()
    notebook = json.loads(notebook_bytes)
    require(set(notebook["metadata"]) == {"kernelspec", "language_info"}, "Unexpected notebook metadata.")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    require(len(code_cells) == 1, "Expected one archived training code cell.")
    cell = code_cells[0]
    require(cell["outputs"] == [] and cell["execution_count"] is None, "Historical outputs must be removed.")
    require(all(c.get("metadata") == {} for c in notebook["cells"]), "Unexpected cell metadata.")
    source = "".join(cell["source"])
    tree = ast.parse(source)
    flags = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "RUN_GPU_TRAINING" for t in n.targets)]
    require(len(flags) == 1 and isinstance(flags[0], ast.Constant) and flags[0].value is False,
            "RUN_GPU_TRAINING must be False in the published notebook.")
    require(source.count("RUN_GPU_TRAINING = False") == 1, "Unexpected training flag spelling.")
    original_cell = source.replace("RUN_GPU_TRAINING = False", "RUN_GPU_TRAINING = True", 1)
    require(digest(original_cell.encode()) == ORIGINAL_CELL_SHA, "Original notebook code was changed beyond its safety flag.")
    require(provenance["notebook"]["original_code_cell_sha256"] == ORIGINAL_CELL_SHA, "Original notebook provenance disagrees.")
    require(digest(source.encode()) == provenance["notebook"]["published_code_cell_sha256"], "Published code provenance disagrees.")
    require(digest(notebook_bytes) == provenance["notebook"]["sha256"], "Published notebook file hash changed.")
    require(provenance["this_import"]["training_started"] is False and provenance["this_import"]["gpu_used"] is False,
            "This must describe an archive import, not a new run.")
    return {"status": "TRAINING_ARCHIVE_PROVENANCE_PASS", "original_python_sources": len(SOURCE_HASHES),
            "source_syntax_valid": True, "configuration_fingerprint_exact": True,
            "serving_model_class_exact": True, "notebook_training_enabled": False,
            "original_notebook_code_reconstructed_exactly": True, "private_weights_required": False,
            "training_started": False, "gpu_used": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
