# Training history and decisions

This is the **28 October timeline task**, completed early on **27 October 2026**. It recovers existing experiment evidence; no training, inference or GPU session was started. Git timestamps retain the real completion date.

The deployed model remains the original **ConvNeXt-Tiny epoch-8 checkpoint**. Its selection can be replayed from the saved eight-epoch log. An earlier EfficientNet attempt stopped before a validated candidate existed; a later extra-epoch label-refinement trial did not improve the original selection metric. This is a partial, evidence-backed history rather than a complete list of every notebook tried.

## Original eight-epoch run

![Original training and weak-validation curves recovered from saved logs](training_curves.png)

Every point comes from the original `training_history.csv`, whose hash matches the [evaluation recovery provenance](epoch8_evaluation_provenance.json). [The full-precision history and experiment ledger](experiment_history.json) and [history provenance](experiment_history_provenance.json) are machine readable. The figure uses matplotlib 3.10.3 from an existing local Python installation; it adds no application dependency.

| Completed epoch | Training soft BCE | Weak-validation soft BCE | Weak macro AUC | Selected epoch after validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.605832 | 0.581714 | 0.776431 | 1 |
| 2 | 0.571159 | 0.575383 | 0.813494 | 2 |
| 3 | 0.553590 | 0.553864 | 0.828545 | 3 |
| 4 | 0.539506 | 0.551194 | 0.836332 | 4 |
| 5 | 0.528291 | 0.542712 | 0.844457 | 5 |
| 6 | 0.519438 | 0.539919 | 0.845660 | 6 |
| 7 | 0.511864 | 0.539926 | 0.848333 | 6 |
| 8 | 0.507309 | 0.537434 | 0.847996 | 8 |

The [original trainer](../training/archive/convnext/convnext_train.py) accepts a new checkpoint only when `validation_loss < best_loss - 0.0001`, after complete validation on all 637 studies. Replaying that rule gives the selected-epoch sequence above and agrees with the checkpoint and run summary. Epoch 7 has the highest weak-validation AUC, but its BCE does not improve on epoch 6. Epoch 8 improves BCE and becomes the final selection. Expert-development AUC and leaderboard scores are not inputs to this rule.

The run ended at its configured **eight-epoch limit**, with patience 3 available but not triggered. This establishes which checkpoint this run selected, not that eight epochs are globally optimal. Validation loss still improved at the end; a differently controlled continuation could behave differently.

Training loss uses randomly sampled windows, intensity augmentation and training-mode dropout, while validation uses all 40 windows in evaluation mode. It is also aggregated as the model changes through the epoch. The train/validation gap therefore cannot be interpreted solely as an estimate of overfitting.

There were **3656 accumulation groups, 3655 successful optimizer updates and one skipped update**. Group 2791 in epoch 7 recorded a nonfinite gradient norm; mixed-precision scaling skipped that update and training continued. It was not a failed run or a fabricated failure example. The saved performance table, cumulative epoch counters and summary all agree.

Historical runtime fields have different scopes: the trainer reports **31.5075 minutes**, while Kaggle displays **32m 11s, GPU T4 x2** for the saved version. The source addresses `cuda:0`; the display does not establish that two GPUs trained the model in parallel. The first 25 successful updates formed the retained benchmark, with five warmup updates excluded and a last-ten median of 0.5828 seconds per update. These are historical observations, not compute consumed by this documentation work.

Source: [Knee ConvNeXt Tiny Budget, version 360434584](https://www.kaggle.com/code/owaiskhan15/knee-convnext-tiny-budget?scriptVersionId=360434584). Training settings, labels and data boundaries are in the [training archive](../training/README.md); prediction-level metric reconciliation is in [per-finding evaluation](PER_FINDING_EVALUATION.md).

## Earlier budget gate: no EfficientNet result to compare

[Knee EfficientNet B0 Budget, version 359488326](https://www.kaggle.com/code/owaiskhan15/knee-efficientnet-b0-budget?scriptVersionId=359488326), recorded `benchmark_viable=False`, `completed_epochs=0` and `candidate_ready=False`. The budget gate stopped the run before it could fit a complete epoch plus validation. The training function reported 3.5601 minutes; Kaggle's saved runtime display is 4m 23s.

There is no selected validation AUC or competition score from this attempt in the recovered record. It would be incorrect to assign it zero AUC, call its model accuracy poor, or conclude that ConvNeXt is inherently faster. Profiling methods, warmup conditions and execution details differed between the attempts; these were not matched encoder benchmarks.

## Later fixed label-refinement attempt

[Knee ConvNeXt Label Refinement, version 361212433](https://www.kaggle.com/code/owaiskhan15/knee-convnext-label-refinement?scriptVersionId=361212433), started from the trained epoch-8 model and completed **one extra epoch / 457 optimizer updates**. The bounded retry stopped before another full epoch would fit. Its recorded training-function time was 6.0811 minutes under an eight-minute cap. This version is a saved snapshot of logged work; its short Quick Version display is not the duration of that training worker.

The attempt changed fitting labels using a fixed JEV-based policy. Where both an original label and an addressed replacement existed, they were blended 50/50; addressed replacements could fill unknown fitting labels. The policy used Pilkwang verdicts for ACL/MCL/Fracture and Steven-v4 source handling for other targets, including that source recipe's exact 0.25 missing marker. That marker is not a general probability convention. Non-fitting rows and study/group assignments were checked unchanged by the policy.

It also lowered encoder/head learning rates from **3e-5 / 3e-4** to **2e-5 / 1e-4**, created a new optimizer, and restarted warmup/cosine scheduling. Extra training, label changes, learning rates and optimizer/schedule changes are confounded. The result does not isolate the effect of JEV labels.

| Model/checkpoint | Weak BCE | Weak AUC | Expert-development AUC | Public score | Private score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original epoch 8, deployed | 0.537434 | 0.847996 | 0.853143 | 0.833 | 0.824 |
| Fixed last checkpoint after one extra epoch, trial only | 0.566695 | 0.848575 | 0.848593 | 0.834 | 0.823 |

The later run's `best_epoch=0` means **the original eight-epoch starting checkpoint was retained**. It does not mean an untrained network. Its summary's selected-checkpoint BCE/AUC and expert AUC therefore describe epoch 8. The `EPOCH_COMPLETE` record gives the actual extra epoch's validation results; a separate saved CPU evaluation gives the fine-tuned last checkpoint's expert AUC. Keeping those identities separate avoids incorrectly reporting the retained model's metrics as refinement results.

The actual last checkpoint has SHA-256 `edff39da7f87ddfa33f11604af313cc1e81bd32f4780b7b93543652b9110f29b`. Its expert result is recovered from [CPU evaluation version 361217016](https://www.kaggle.com/code/owaiskhan15/knee-convnext-epoch9-cpu-evaluation?scriptVersionId=361217016), not recalculated from trial prediction tables during this recovery. The [submitted inference version 361219648](https://www.kaggle.com/code/owaiskhan15/knee-convnext-epoch9-cpu-evaluation?scriptVersionId=361219648) identifies the same checkpoint and explicitly says it is not the BCE winner.

Paired public/private scores were read from the [competition submissions table](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions) on 27 October. The original submission is [version 360716321](https://www.kaggle.com/code/owaiskhan15/knee-convnext-verified-inference?scriptVersionId=360716321). The trial's 0.001 public increase did not carry through to its private score; these rounded differences are descriptive and do not establish statistical significance. The notebook's standalone score badge currently shows the **private** score, so it must not be mistaken for the historical public score. Private scores are recorded retrospectively; this work changes no final competition selections.

The refinement's training loss of 0.485356 uses changed targets and cannot be directly compared with the original 0.507309 as proof of better learning. Weak AUC rose slightly while the unchanged validation BCE worsened. The expert result also declined, but expert58 was repeatedly reused and the JEV label source used it in development; neither expert result is independent test performance. The original selection rule already rejected the candidate, so the deployed model stays unchanged.

## Evidence gaps and what this teaches

Earlier ResNet18/DINO experiments and refinement versions 1–4 are not fully reconstructed here. Their full logs, configurations, checkpoints or matched prediction tables are not all available in this archive. No missing curves, failed-run scores, total GPU-hour accounting or complete causal comparison is supplied for them. The separate public-model Raptor/S6 reproduction is not this independently trained classifier and is not used as an encoder-ablation result.

The supported lessons are to budget against a complete validation cycle, preserve exact checkpoint identities, separate soft-label loss from ranking metrics, and test one hypothesis at a time. A small leaderboard movement can disagree with validation and private results. A future controlled experiment must keep its split, input windows, initialization, optimizer/schedule and compute budget fixed while changing one intended factor. This history does not add that future experiment's results.

## Rebuild the record without training

Keep the nine original private files listed in [history provenance](experiment_history_provenance.json) together under ignored `data/private_history/`. The CLI verifies every file hash, cross-checks the original history/summary with the earlier evaluation recovery, replays checkpoint selection and optimizer counters, and reads only explicit saved output markers from notebook JSON. It never executes notebook cells or imports a model.

```powershell
python -S scripts/build_experiment_history.py --evidence-dir data/private_history --output .runtime/history_rebuilt.json
python -m pytest -q tests/test_experiment_history.py
```

The first command must print `HISTORY_RECONCILED`. Its aggregate JSON should match `docs/experiment_history.json` exactly. A public clone can run the artificial-data tests without private files or weights. To render the figure in a separate environment where matplotlib is already installed, omit `-S` and add `--plot .runtime/training_curves.png`; the core reconstruction does not need matplotlib. Figure bytes can depend on plotting/font versions; the underlying values are exact in the JSON.

Raw notebook outputs and all study-level data remain private. The public chart contains only aggregate epoch metrics. No serving source, checkpoint, dependency lockfile, model selection, paid service or hosting configuration changed.
