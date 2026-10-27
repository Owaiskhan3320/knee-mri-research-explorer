# Reproduced epoch-8 evaluation

On **27 October 2026**, the original saved ConvNeXt-Tiny predictions were recovered and evaluated again on CPU, without loading a model or running inference. All 12 per-finding AUCs and both macro AUCs reconcile with the original metric tables and run summary. This work documents the existing model; its public competition score remains **0.833**.

| Evaluation | Studies | Known finding labels | Recomputed macro ROC AUC | Purpose |
| --- | ---: | ---: | ---: | --- |
| Weak validation | 637 | 5563 / 7644 | 0.847995649271 | Used for checkpoint selection by masked soft-label BCE |
| Expert development | 58 | 696 / 696 | 0.853143219751 | Repeatedly used development diagnostic |

The expert set is small and has been reused during development. These figures are not independent test performance, clinical validation, or classification accuracy. Weak-validation AUC measures agreement with report-derived binary verdicts; the expert set uses expert binary labels. Their different labels and study populations prevent a direct interpretation of the difference as an improvement or decline in generalization.

## Per-finding results

All 58 expert-development labels are known for every finding. The weak-validation columns below show how many of the 637 studies actually contribute to each AUC. Negative counts and per-finding soft-label BCE are also available in the aggregate JSON reports.

| Finding | Weak known | Weak positives | Weak AUC | Expert positives | Expert AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL | 586 | 181 | 0.830871 | 24 | 0.870098 |
| MCL | 584 | 121 | 0.766560 | 9 | 0.804989 |
| Medial Meniscus | 605 | 370 | 0.837723 | 26 | 0.813702 |
| Lateral Meniscus | 580 | 172 | 0.787976 | 23 | 0.792547 |
| Medial OA | 476 | 246 | 0.909084 | 15 | 0.948837 |
| Lateral OA | 430 | 174 | 0.871139 | 11 | 0.794971 |
| PF OA | 525 | 304 | 0.893531 | 21 | 0.826255 |
| Effusion | 569 | 381 | 0.867384 | 35 | 0.921739 |
| Synovitis | 99 | 86 | 0.826476 | 27 | 0.837515 |
| Baker's | 331 | 155 | 0.865396 | 12 | 0.911232 |
| Contusion | 507 | 188 | 0.821450 | 19 | 0.838057 |
| Fracture | 271 | 46 | 0.898357 | 18 | 0.877778 |

MCL and lateral meniscus have the lowest weak-validation AUCs. Synovitis has only 99 known weak-validation labels, including 13 negatives; its apparent performance therefore rests on much less evidence than the study count alone suggests. These observations identify questions for later label/error analysis. They do not establish that a particular architectural change will improve the model.

## Evaluation contract

The [offline evaluator](../scripts/evaluate_predictions.py) uses only Python's standard library:

- The label/split table must match the SHA-256 stored in the epoch-8 checkpoint configuration. The prediction files, saved AUC tables and run summary must match the recovered historical output hashes.
- Study IDs are unique, nonempty and matched by name. The predicted study set must exactly equal the requested split; missing or extra studies fail. Group assignments cannot cross splits.
- Prediction columns must be the study ID and the 12 explicitly named findings. Column and row reordering is safe because alignment uses names and IDs; output uses the checkpoint's canonical target order.
- All predictions must be finite probabilities in `[0, 1]`, including predictions whose labels are unknown. Blank/NaN labels are masked. Soft-label and binary-label known masks must agree. Infinite or out-of-range labels fail.
- BCE uses the original soft labels. AUC uses the explicit `__binary` columns: weak labels come from YES/NO verdicts, while expert rows use expert labels. The evaluator never thresholds soft labels or substitutes confidence values.
- ROC AUC uses a rank-equivalent Mann–Whitney calculation, with half credit for ties. Unknown labels contribute neither positive nor negative examples. A single-class finding has `auc=null` and an explicit reason; it is excluded from the macro mean, whose denominator is reported. All 12 AUCs are defined in these actual evaluations.
- Soft-label BCE is summed over all known study–finding entries, then divided by their count. This is different from giving each finding or study equal BCE weight. BCE reconstructed from saved sigmoid probabilities can differ slightly from the trainer's float32 logit loss. Endpoint probabilities are rejected for this reconciliation rather than silently clipped. Neither recovered prediction table contains endpoints.

Per-finding and macro AUC comparisons use an absolute tolerance of **1e-12**; the weak-validation BCE comparison uses **1e-6**. Maximum AUC differences are **2.22e-16** for weak validation and **1.11e-16** for expert development. Recomputed weak soft BCE is **0.537434014871**, versus original logit BCE **0.537434004889**, a difference of **9.98e-9**. No tolerance was widened after a failed comparison.

Epoch 8 remains selected by the original complete weak-validation BCE rule. This evaluation does not reselect a checkpoint using expert results or AUC.

## Reproduce with authorized private outputs

The [training notebook, version 360434584](https://www.kaggle.com/code/owaiskhan15/knee-convnext-tiny-budget?scriptVersionId=360434584), contains `val_predictions.csv`, `gold_predictions.csv`, `val_auc.csv`, `gold_auc.csv` and `run_summary.json` under `knee_convnext_tiny/`. The exact `study_labels_splits.csv` comes from the [CPU-preparation notebook, version 360433261](https://www.kaggle.com/code/owaiskhan15/knee-convnext-cpu-preparation?scriptVersionId=360433261), under `knee_convnext_preparation/bundle/`.

The deployed checkpoint inside the saved training output archive was also SHA-256 checked against the deployed model identity. The label table matches the checkpoint-bound hash. [Recovery provenance](epoch8_evaluation_provenance.json) records file sizes and hashes. No training session was started to recover these files.

Keep the seven recovered files together under ignored `data/private_evaluation/`. With Python 3.12, run from the repository root:

```powershell
python -S scripts/evaluate_predictions.py --predictions data/private_evaluation/val_predictions.csv --labels data/private_evaluation/study_labels_splits.csv --split weak_validation --recorded-auc data/private_evaluation/val_auc.csv --recorded-summary data/private_evaluation/run_summary.json --output .runtime/evaluation/weak_validation.json

python -S scripts/evaluate_predictions.py --predictions data/private_evaluation/gold_predictions.csv --labels data/private_evaluation/study_labels_splits.csv --split gold --recorded-auc data/private_evaluation/gold_auc.csv --recorded-summary data/private_evaluation/run_summary.json --output .runtime/evaluation/expert_development.json
```

Both commands must report `RECONCILED`. The checked-in results are [weak validation](epoch8_weak_validation.json) and [expert development](epoch8_expert_development.json). The CLI writes only aggregates, provenance hashes and fixed source links. It does not include study IDs, groups, individual labels/predictions or local input paths in its output.

A public clone can run the evaluator's artificial-data tests without weights, medical data, training libraries or network access:

```powershell
python -m pytest -q tests/test_evaluate_predictions.py
```

Tests cover hand-calculated and exhaustive-pair AUCs with ties, unknown masking, soft-label BCE, row/column alignment, incomplete/duplicate studies, invalid probabilities, group leakage, single-class/no-label handling, duplicate CSV headers, changed historical hashes, and the complete CSV-to-report command path. Full numerical reproduction of the historical scores requires authorized access to the private inputs. They are intentionally not distributed in this repository.

The [training history and experiment record](TRAINING_HISTORY.md) includes the recovered curves and earlier/later attempts. [Label coverage and error analysis](ERROR_ANALYSIS.md) adds fixed-cutoff disagreements, ranking-error counts and a private review queue. These records add no threshold tuning, confidence intervals or clinical claim.
