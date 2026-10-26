"""Create a fresh CPU-only ConvNeXt bundle using verified unchanged V1 labels."""
import gc
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def prepare(output, parent_bundle=None):
    assert not torch.cuda.is_available(), "Set accelerator None; preparation must use CPU"
    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    output = Path(output)
    bundle = output / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    module_path = bundle / "cnn_convnext.py"
    import hashlib
    module_name = "knee_convnext_model_" + hashlib.sha256(module_path.read_bytes()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.write_json(dict(complete=False, gpu_used=False), output / "preparation_status.json")
    parent = Path(parent_bundle or module.DEFAULT_PARENT_BUNDLE)
    _, old_cfg, frame, pixels = module.base.load_bundle(parent, require_preflight=True)
    assert frame.groupby("split").size().to_dict() == {"fit": 3655, "weak_validation": 637, "gold": 58}
    assert frame.groupby("verified_group").split.nunique().max() == 1
    for name in ["study_labels_splits.csv", "pixel_index.csv"]:
        shutil.copyfile(parent / name, bundle / name)
        assert module.sha256(parent / name) == module.sha256(bundle / name)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    assert weights.url == "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
    print("Downloading official torchvision ConvNeXt-Tiny initialization on CPU:", weights.url, flush=True)
    official_encoder = convnext_tiny(weights=weights)
    encoder_state = official_encoder.state_dict()
    model = module.StudyMIL(num_targets=12, slots=5, projection=256, encoder_state=encoder_state)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(isinstance(item, torch.nn.modules.batchnorm._BatchNorm) for item in model.encoder.modules())
    module.atomic_torch(model.state_dict(), bundle / "initial_model.pt")
    encoder_parameter_count = sum(parameter.numel() for parameter in model.encoder.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del official_encoder, encoder_state, model
    gc.collect()
    cfg = dict(targets=old_cfg["targets"], slots=old_cfg["slots"], seed=42,
               pixel_key=old_cfg["pixel_key"], feature_key=old_cfg.get("feature_key"),
               data_hashes=old_cfg["data_hashes"], pixel_partitions=old_cfg["pixel_partitions"],
               expected_counts=old_cfg["expected_counts"], encoder="torchvision.convnext_tiny",
               pretrained_weights=str(weights), official_weight_url=weights.url,
               encoder_parameter_count=encoder_parameter_count, parameter_count=parameter_count,
               parent_bundle=str(parent), parent_config_sha256=module.sha256(parent / "config.json"),
               parent_module_sha256=module.PARENT_MODULE_SHA256,
               max_gpu_minutes=240,
               label_source=old_cfg["label_source"],
               selection="minimum_full_weak_validation_soft_bce",
               input_contract="Existing unchanged 224px cached adjacent-slice triplets; ImageNet normalization; no extra resize/crop",
               expert_role="Repeatedly used development diagnostic; not independent validation",
               settings=dict(projection=256, train_windows_per_slot=2, validation_windows=40,
                             batch_size=2, accumulation=4, workers=2, epochs=8,
                             encoder_lr=3e-5, head_lr=3e-4, weight_decay=1e-4,
                             patience=3, min_delta=1e-4, warmup_steps=40,
                             intensity_jitter=.1, encoder_chunk=32,
                             max_minutes=240,
                             benchmark_warmup_optimizer_updates=5,
                             benchmark_steady_optimizer_updates=20,
                             validation_probe_warmup_batches=2,
                             validation_probe_timed_batches=8,
                             cudnn_benchmark=False, runtime_benchmark_required=True,
                             execution_settings_require_gpu_profile=False),
               runtime=dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                            python=sys.version.split()[0]),
               hashes={name: module.sha256(bundle / name) for name in
                       ["cnn_convnext.py", "parent_cnn_training.py", "prepare_convnext.py", "convnext_train.py",
                        "study_labels_splits.csv", "pixel_index.csv", "initial_model.pt"]})
    module.write_json(cfg, bundle / "config.json")
    print("Unchanged split verified; running CPU backward and full-bag mask checks.", flush=True)
    preflight = module.cpu_preflight(bundle)
    module.write_json(preflight, bundle / "preflight.json")
    cfg["hashes"]["preflight.json"] = module.sha256(bundle / "preflight.json")
    module.write_json(cfg, bundle / "config.json")
    module.load_bundle(bundle, require_preflight=True)
    summary = dict(complete=True, gpu_used=False, training_started=False,
                   training_studies=3655, validation_studies=637, expert_development_studies=58,
                   pixel_manifest_studies=len(pixels), training_labels_changed=False,
                   original_v1_bundle_unchanged=True, cpu_preflight_passed=True,
                   encoder="torchvision.convnext_tiny", encoder_trainable=True,
                   official_weight_url=weights.url, parameter_count=parameter_count,
                   encoder_parameter_count=encoder_parameter_count,
                   cpu_minutes=(time.monotonic() - started) / 60,
                   data_fingerprint=module.fingerprint(cfg),
                   next_step="One GPU run with retained-update ConvNeXt benchmark and mandatory budget gate before continuing",
                   submission_created=False)
    module.write_json(summary, output / "preparation_status.json")
    print("CONVNEXT_CPU_PREPARATION_COMPLETE", json.dumps(summary, indent=2), flush=True)
    return summary
