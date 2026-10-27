# Pooling experiment feasibility decision

This is the **31 October timeline task**, completed early on **27 October 2026**. The decision is **defer execution because prerequisites are not ready**. The [paired pooling protocol](POOLING_EXPERIMENT.md) remains disabled, and no experiment or accuracy comparison was run.

This records a concrete readiness check, not a conclusion that CPU training is inherently impossible. Without the complete fixed inputs and a tested CPU runner, the proposed 120-minute / 4-GiB limit cannot be measured or declared feasible. Timing, peak training RSS and experiment results therefore remain null in the [aggregate audit](pooling_feasibility.json).

## What was checked

The audit inspected filenames under the project's ignored `data/`, `models/` and `.runtime/` directories, excluding environment/dependency directories. It also inspected central-directory entries in the seven already downloaded `results*.zip` archives associated with this project. It did not search every drive, inspect remote Kaggle storage, recursively expand nested archives, extract files, or download data.

The original preparation archive supplied a hash-verified `pixel_index.csv`. Its 4350 unique study rows refer to **two partitions of 2175 studies each** and **88 partition/shard references**. There are 44 distinct shard basenames because the two partitions reuse names such as `pixels_000.zip` in separate roots.

| Prerequisite | Observed evidence | Decision |
| --- | --- | --- |
| Fixed label/split table | 4350 studies; exact checkpoint-bound SHA-256 match | Available |
| Pixel index | 1,274,599 bytes in the saved preparation archive; exact SHA-256 match | Available |
| Original pretraining initial state | 112,429,834 bytes in the same archive; streamed SHA-256 matches | Available in archive; not deserialized |
| Complete image-cache shards | No matching shard filename candidates in the audited directories or archive entries; no local pixel manifest/status candidates | Not verified or located in the audited scope |
| Existing replay inputs | Three unique studies; reference-input hashes match; all three are weak validation | Not a fitting dataset |
| CPU mean-pooling model/runner and model tests | Not implemented; the archived training driver requires CUDA | Pending |
| Representative paired CPU timing and training RSS | No benchmark started after the readiness failure | Unmeasured |

The private replay-to-split join found **0 fitting, 3 weak-validation and 0 expert-development studies**. Their IDs remain private. Three implementation-parity examples cannot replace the protocol's 3655 fitting studies or 637-study validation set. The index and initialization file contain metadata and model state, not the missing image shards.

The audit's scope matters: absence of matching filenames in these locations does not prove that no suitable data exist elsewhere. It does establish that the currently audited project assets do not provide the verified complete inputs required by this experiment. Any later restoration needs the original partition manifests and packed-member hashes checked, rather than merely finding similarly named ZIP files.

## Why execution stopped here

The protocol requires verified complete inputs, a tested CPU implementation, and representative full-group/full-window profiling before fitting. The data gate failed first. Porting a trainer or timing a smaller substitute would not establish readiness for the specified comparison, so no model construction, optimizer step, inference or resource benchmark was attempted.

The two-arm design still requires the same original initialization, labels, sample order, windows, eight complete epochs and BCE-based checkpoint selection. There is no shortened dataset, unequal-epoch comparison, trained-checkpoint pooling swap or frozen-feature substitute. The original deployed ConvNeXt model, its scores and competition selections remain unchanged.

There is **no experiment-results commit** for this task. The published change contains this feasibility decision and aggregate evidence only. It does not claim that the pooling comparison failed on accuracy, exceeded its measured budget, or produced a new score.

## Evidence identities

The audit binds to protocol SHA-256 `0c383efd373669309c3bbb6c06e01fceb805b60bd48f85ccf8147c6479b962d2`. The [machine-readable record](pooling_feasibility.json) includes source hashes, sizes, inspected scope, readiness flags and null measurements.

The available input hashes match the original checkpoint configuration:

| Asset | SHA-256 |
| --- | --- |
| Label/split table | `82401171a2bade3e7a18aa226d453a61e14e1c3591c26f2a669fa7cd2ed0eb99` |
| Pixel index | `a7c3920ba0e91aee6a18b86aa1ca888b43db61ba06ccfcce7e3781a435cb985e` |
| Initial state | `bf6c454ddb7f1ed055f4fbeaca51a46b655444d6a40108e725a7795f4ab17da9` |

These identities come from the [saved CPU preparation, version 360433261](https://www.kaggle.com/code/owaiskhan15/knee-convnext-cpu-preparation?scriptVersionId=360433261), and the archived [checkpoint configuration](../training/config/epoch8_config.json). Initial-state hashing reads bytes without loading tensor objects or constructing a model. The private operator audit is retained under ignored `.runtime/`; no study-level evidence is published.

For a later readiness audit, verify that the protocol hash still matches, verify the label/index/initial-state hashes, enumerate the exact `(partition, shard)` references in the index, locate and validate the original cache manifests and every referenced packed member, and repeat the private replay-to-split accounting. A future asset survey is a new snapshot; this report is not a live filesystem monitor or a claim of identical readiness on every machine.

## What remains

The pooling experiment can be reconsidered after authorized complete cache inputs are restored, both CPU arms/runner pass the planned correctness tests, and full paired profiling establishes compliance with the resource ceilings. Zero GPU hours and no paid services remain the current constraints. No automatic download, retry, resume, budget extension or background training has been configured.

The next portfolio task is **1 November: consolidate the README around the problem, model, verified results and lessons**, linking this deferral instead of presenting an unrun ablation as completed experimental work. Training can remain deferred while the documentation and release tasks proceed.
