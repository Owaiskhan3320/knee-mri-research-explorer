# Credits and data boundaries

## Project work

Owais Khan's project trained the selected ConvNeXt-Tiny attention-MIL model using ImageNet initialization and publicly supplied weak labels. Training implementation and the local application, preprocessing integration, model packaging, input checks, worker/API/UI and verification tooling used AI coding assistance. The architecture and training data are not claimed as original research inventions. The [training archive](training/README.md) preserves the original source/configuration and records its recovery from saved Kaggle artifacts.

The deployed checkpoint is the project's epoch-8 model, confirmed public macro ROC AUC **0.833**. Public Raptor/S6 models and the separate 0.950 reproduction are not part of this application's model.

## MRI data

- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection): competition MRI data, series descriptors and expert annotations, supplied by RSNA and contributing institutions.
- [Competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) and [RSNA MIRA research-use agreement](https://www.rsna.org/mira-license): consult the applicable terms before accessing or reusing data. The MIRA agreement restricts redistribution and clinical use.

No competition scans, reports, label tables, reference input arrays, study identifiers or trained weights are distributed with this code repository. The checked-in screenshot/video use generated mathematical phantoms, not patient images. The MIT license below does not grant rights in MRI data or weights.

## Report-derived training labels

[Pilkwang Kim's **RSNA Knee LLM-read report labels**](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), dataset version 1, `report_labels_v2.csv`. The dataset page identifies its license as CC0: Public Domain (checked 25 October 2026).

The project consumed these already published labels. It did not build a new multilingual report extraction system or call a paid label-generation API. The training pipeline retained soft per-finding scores for `YES`/`NO` verdicts and masked `UNK` findings. Although confidence columns were checked for schema compatibility, this baseline did not claim a confidence-weighted label policy. Later JEV label-refinement experiments are not deployed here.

## Model and learning methods

- [ConvNeXt — A ConvNet for the 2020s, Liu et al.](https://arxiv.org/abs/2201.03545): backbone architecture.
- [torchvision ConvNeXt-Tiny implementation and ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html): pretrained initialization for training. Deployment loads only the pinned project checkpoint, with no download at startup.
- [Attention-based Deep Multiple Instance Learning, Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html): background for gated attention pooling. This project's finding-specific pooling is an implementation choice, not a claim to invent MIL.

## Software

PyTorch/torchvision, pydicom and pylibjpeg plugins, NumPy, Pillow, pandas, FastAPI/Starlette/Uvicorn, Gradio, httpx and pytest retain their respective licenses. Their dependencies are installed from official package registries and are not vendored in this source release. Version pins are in requirements-lock.txt; PyTorch CPU pins are in setup.ps1 and Dockerfile.

The repository's MIT license covers project-authored source and documentation only. It does not replace third-party licenses, grant dataset access or authorize redistribution of trained weights. Model distribution remains a separate review item.
