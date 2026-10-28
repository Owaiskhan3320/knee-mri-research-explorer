# Original ConvNeXt-Tiny training

This archive makes the training behind the deployed epoch-8 model inspectable. It was imported on **26 October 2026** from the saved Kaggle experiment and its pinned inputs. No training was rerun for this import.

The project trained a ConvNeXt-Tiny encoder and attention-MIL head using ImageNet initialization and Pilkwang Kim's published report-derived labels. Its public competition macro ROC AUC is **0.833**. Training and application development used AI coding assistance; the backbone, MIL method and published labels are credited in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Kaggle experiment timeline

These are selected historical notebook versions, verified in Kaggle's **Version History on 29 October 2026**. Dates are the version-save dates as Kaggle displayed them; that panel does not label a timezone. Each selected version was checked against its `Edit/run` version ID. [Exact displayed times and evidence identities](../docs/kaggle_timeline.json).

| Saved date (Kaggle display) | Notebook and exact version | Save type | What it records |
| --- | --- | --- | --- |
| 18 Oct 2026 | [EfficientNet-B0 budget attempt, V1 / 359488326](https://www.kaggle.com/code/owaiskhan15/knee-efficientnet-b0-budget?scriptVersionId=359488326) | Save & Run All | Budget gate stopped before a validated candidate |
| 20 Oct 2026 | [ConvNeXt CPU preparation, V1 / 360433261](https://www.kaggle.com/code/owaiskhan15/knee-convnext-cpu-preparation?scriptVersionId=360433261) | Save & Run All | Fixed inputs, initialization and preparation checks |
| 20 Oct 2026 | [ConvNeXt-Tiny training, V1 / 360434584](https://www.kaggle.com/code/owaiskhan15/knee-convnext-tiny-budget?scriptVersionId=360434584) | Save & Run All | Eight completed epochs; BCE selected epoch 8 |
| 21 Oct 2026 | [Raw/cache parity, V1 / 360713817](https://www.kaggle.com/code/owaiskhan15/knee-convnext-raw-parity?scriptVersionId=360713817) | Save & Run All | Verified CPU preprocessing/inference bundle |
| 21 Oct 2026 | [Verified inference, V1 / 360716321](https://www.kaggle.com/code/owaiskhan15/knee-convnext-verified-inference?scriptVersionId=360716321) | Save & Run All | Submitted epoch-8 model: 0.833 public / 0.824 private AUC |
| 22 Oct 2026 | [Label refinement, V5 / 361212433](https://www.kaggle.com/code/owaiskhan15/knee-convnext-label-refinement?scriptVersionId=361212433) | Quick Version | Saved logs from the extra-epoch trial; base checkpoint retained by BCE |
| 22 Oct 2026 | [Trial expert check, V1 / 361217016](https://www.kaggle.com/code/owaiskhan15/knee-convnext-epoch9-cpu-evaluation?scriptVersionId=361217016) | Quick Version | Saved expert-evaluation and pixel-parity evidence for the trial |
| 22 Oct 2026 | [Trial inference snapshot, V3 / 361219648](https://www.kaggle.com/code/owaiskhan15/knee-convnext-epoch9-cpu-evaluation?scriptVersionId=361219648) | Quick Version | Submitted trial checkpoint: 0.834 public / 0.823 private AUC |
| 24 Oct 2026 | [First local replay export, V1 / 361757109](https://www.kaggle.com/code/owaiskhan15/knee-local-windows-replay-export?scriptVersionId=361757109) | Save & Run All | Private reference inputs for Windows/Linux verification |
| 25 Oct 2026 | [Additional replay export, V1 / 362222429](https://www.kaggle.com/code/owaiskhan15/knee-additional-local-replay-export?scriptVersionId=362222429) | Save & Run All | Two further private cases, extending parity coverage to three studies |

**Save dates and execution dates are different evidence.** Quick Version entries date saved snapshots, not the start or duration of their logged workers. Hash-checked original notebook metadata records execution windows of **17 Oct, 21:01-21:06 UTC** for EfficientNet and **20 Oct, 13:31-14:03 UTC** for ConvNeXt. The machine-readable timeline preserves the full UTC timestamps. No missing execution dates were inferred from relative labels such as “days ago.”

The GitHub app/source history starts on **25 October**, this training archive was imported on **26 October**, saved-result reconstruction followed on **27 October**, and the release review was recorded on **28 October**. Those commits document publication and recovery of earlier experiments; they are not new training runs. The timeline above is a selected record, not a reconstruction of every earlier ResNet/DINO notebook or data-preparation step.

## Start reading here

| File | Role in the original pipeline |
| --- | --- |
| [Original notebook entry point](notebooks/convnext_tiny_budget.ipynb) | Validates the prepared bundle and explicitly calls the ConvNeXt trainer. Published with training disabled and outputs removed. |
| [convnext_train.py](archive/convnext/convnext_train.py) | Actual GPU training loop: accumulation, mixed precision, schedule, budget checks, validation and checkpoint selection. |
| [cnn_convnext.py](archive/convnext/cnn_convnext.py) | ConvNeXt model, finding-specific gated attention and verified access to the shared cache/loss runtime. |
| [parent_cnn_training.py](archive/convnext/parent_cnn_training.py) | Shared packed-image reader, masked loss and metrics. It also contains an earlier EfficientNet model/driver, which the ConvNeXt notebook does not call. |
| [prepare_convnext.py](archive/convnext/prepare_convnext.py) | Copies the verified split/cache index, creates ImageNet initialization and runs CPU preparation checks. |
| [Parent preparation](archive/parent_preparation/prepare.py) and [label audit](archive/parent_preparation/label_audit.py) | Verify inherited labels/splits and cache provenance before ConvNeXt preparation. The parent runtime is also retained under its original `cnn_training.py` filename. |
| [Checkpoint configuration](config/epoch8_config.json) | Complete `config` dictionary recovered from the verified deployed checkpoint, without model tensors. |
| [provenance.json](provenance.json) | Source hashes, recovery locations, original version links and the notebook's documented safety edit. |

The seven archived Python files retain their original bytes. The notebook has one code change: `RUN_GPU_TRAINING = False`. Its outputs, execution counts and original metadata were removed, and an archive note was added. The provenance verifier can reconstruct and hash-check the original code cell by reversing that one flag change.

## Original experiment settings

| Setting | Recorded value |
| --- | --- |
| Encoder initialization | torchvision ConvNeXt-Tiny, ImageNet-1K V1; encoder trainable |
| Random seed | 42 |
| Maximum epochs / selected epoch | 8 / 8 |
| Input | Five plane/fat-suppression slots; three neighbouring slices per window; 224×224 pixels |
| Training sampling | Two windows per slot before missing-window masking |
| Validation input | All eight windows per slot, 40 total before masking |
| Batch size / gradient accumulation | 2 / 4 |
| Optimizer | AdamW; encoder LR 0.00003, head LR 0.0003; weight decay 0.0001 |
| Schedule | 40-update warmup, then cosine decay with a 0.1 multiplier floor |
| Loss | Soft-label BCE on known targets, normalized over known entries in the accumulation group |
| Augmentation | Study-level intensity gain jitter of ±0.1 |
| Other settings | Projection 256; projection dropout 0.2; gradient clipping at 5; CUDA FP16 autocast/GradScaler |
| Early stopping | Patience 3, validation-loss improvement threshold 0.0001 |

The saved `max_gpu_minutes=240` is the original configured ceiling. Kaggle displays **32m 11s, GPU T4 x2** for the historical notebook version; these describe different quantities. They do not represent resources used by this archive import. The trainer addresses `cuda:0`.

The configuration's `runtime` field was written during **CPU preparation**. It is not a complete record of the GPU training environment, and the serving environment's lockfile should not be presented as an original training lockfile.

## Splits, labels and selection

| Split | Studies | Purpose |
| --- | ---: | --- |
| `fit` | 3655 | Optimize the model using report-derived soft labels |
| `weak_validation` | 637 | Full validation loss and checkpoint selection |
| `gold` | 58 | Repeatedly used expert development diagnostic |

These assignments were inherited from the earlier baseline/DINO data preparation. The parent preparation preserves them, renames `val` to `weak_validation`, and checks that every `verified_group` stays within one split. ConvNeXt preparation copies the split table and pixel index byte for byte. This import does not reconstruct the earlier group/split-generation implementation or independently re-audit the private tables.

Only finite Pilkwang scores with `YES`/`NO` verdicts are retained as weak soft targets; `UNK` is masked even when a numeric score is present. Weak-validation binary labels used for AUC come from the verdict. Confidence columns are not used as class targets or loss weights. The archived label audit compares other published sources descriptively; it does not substitute them into this training run.

The trainer selects a checkpoint only after complete 637-study validation, when masked soft-label BCE improves by more than `min_delta=0.0001`. The selected epoch is 8. The selected checkpoint is frozen before the expert-development prediction step. The 58-study result is not independent test performance. The public leaderboard is not the checkpoint-selection rule encoded in this trainer.

## Verify the archive without training dependencies

From the repository root:

```powershell
python -S scripts/verify_training_archive.py
```

This standard-library check verifies original source hashes, the checkpoint configuration fingerprint, model-class agreement with the serving code, and the notebook's disabled training flag. It does not import the archived modules, PyTorch or any data. No weights are needed for this check.

## Original execution and remaining dependencies

- [Training notebook, version 360434584](https://www.kaggle.com/code/owaiskhan15/knee-convnext-tiny-budget?scriptVersionId=360434584)
- [Saved CPU parity bundle, version 360713817](https://www.kaggle.com/code/owaiskhan15/knee-convnext-raw-parity?scriptVersionId=360713817)

Original execution requires authorized access to the pinned Kaggle inputs: fixed study/group/label tables, both image-cache partitions, hashed CPU-preparation bundles and initialization weights. Their paths and identities are retained in the configuration and sources; their contents are excluded from this repository. The earlier baseline/DINO split and cache-generation entry points are referenced dependencies, not a complete public reconstruction here.

The archive is an inspection record of the earlier experiment. App startup, setup and archive verification do not invoke its preparation or training functions. Calling the preparation functions can download official initialization weights and create private data artifacts; this import did neither. A future training run would need its own compute plan.

On 27 October, [per-finding evaluation](../docs/PER_FINDING_EVALUATION.md) was reproduced from the recovered original predictions and exact checkpoint-bound label table. Both macro AUCs and all 12 finding AUCs reconcile with the saved run; only aggregate results and provenance are published. Full reproduction still requires authorized private inputs. The [training history](../docs/TRAINING_HISTORY.md) also recovers the real epoch curves, replays checkpoint selection and documents the earlier budget gate and later rejected refinement, with missing evidence and confounded comparisons identified.

A [controlled pooling comparison](../docs/POOLING_EXPERIMENT.md) is specified with fixed shared initialization, inputs, sampling, loss and selection. Its protocol is checked without model/data imports. The [feasibility audit](../docs/POOLING_FEASIBILITY.md) records missing complete local cache inputs and pending CPU implementation/timing, so the two-arm experiment remains unrun and disabled.
