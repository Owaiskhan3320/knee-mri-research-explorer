"""Public clones can test admission/API logic without redistributing weights."""
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption("--require-model", action="store_true", help="Fail if the private model checkpoint is missing")


def pytest_collection_modifyitems(config, items):
    checkpoint = Path(__file__).resolve().parents[1] / "models" / "deploy.pt"
    if checkpoint.is_file():
        return  # A wrong/corrupt existing checkpoint must fail the identity/inference tests.
    if config.getoption("--require-model"):
        raise pytest.UsageError("Private model missing: import an authorized bundle before --require-model.")
    missing = pytest.mark.skip(reason="Private model not distributed; import authorized weights or use --require-model.")
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(missing)
