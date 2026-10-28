# Evidence update and final release review

**Review passed on 28 October 2026.** This completes the scheduled **2 November** task early, with genuine Git timestamps. The reviewed source is commit `b7d9aea5496d2783a41025178859dac26471e967`; this review record is a separate documentation change. No application or test-code fix was needed.

This is the evidence/documentation update since the existing `v0.1.0` source release. It does not introduce a new model, application version or release tag. The deployed classifier remains ConvNeXt-Tiny epoch 8: **0.833 public / 0.824 private competition macro ROC AUC**. [Machine-readable review](release_review.json).

## Changes since v0.1.0

- Verified local Linux container operation and expanded exact Windows/Linux input parity to three authorized studies / 464 provided DICOM files. [Deployment evidence](MULTI_STUDY_VERIFICATION.md).
- Archived the original training source, checkpoint configuration, saved-version links and source hashes, with training disabled in the published notebook. [Training archive](../training/README.md).
- Reproduced all per-finding and macro AUCs from original saved predictions and the exact label/split table. [Evaluation](PER_FINDING_EVALUATION.md).
- Recovered real training curves, checkpoint selection and the budget-stopped/rejected attempts; documented their missing evidence and confounded comparisons. [Training history](TRAINING_HISTORY.md).
- Added aggregate supervision coverage and prediction disagreements, keeping the case-review queue private. [Error analysis](ERROR_ANALYSIS.md).
- Specified a controlled pooling comparison and recorded its readiness failure. It remains unrun. [Protocol](POOLING_EXPERIMENT.md) and [feasibility decision](POOLING_FEASIBILITY.md).
- Consolidated the [project overview](../README.md) and separated [local usage](LOCAL_USAGE.md) from the research case study.

The application source, Dockerfile/build context, setup and dependency manifests match the previously verified container baseline `3a9ccac`. The later evidence changes do not alter that serving implementation. Docker was not rebuilt or rerun for this review; its existing results retain their original dates and scope.

## Verification performed

| Check | Result and scope |
| --- | --- |
| Complete suite with authorized private weights | **70 passed, 0 skipped**, using `pytest -q --require-model` on local CPU |
| Clean public GitHub clone without weights/data | **67 passed, 3 skipped**, using the same already installed Python environment |
| Missing-model guard in that clone | `--require-model --collect-only -q` rejected missing weights with exit code 4 |
| Dependencies | `pip check`: no broken requirements |
| Training archive | Original source/configuration/model-class identities pass; published training flag remains disabled |
| Pooling protocol | Valid, disabled and deferred; no candidate or score produced |
| Saved-evidence reproduction | Four aggregate reports rebuilt with standard-library-only commands and matched the published bytes exactly |
| Documentation and source publication | Local file links/heading anchors resolve; code fences close; source/media allowlist and private-file exclusion checks pass |

The complete suite includes the three model-dependent checks: checkpoint/preprocessor identity, masked-window/request independence, and worker timeout/recovery with a successful generated-data inference afterward. These consume local CPU and do not train a model. Public tests use artificial data; the public clone received no private data or weights.

The clean clone was fetched from the existing public GitHub repository at the reviewed commit. **Its dependency environment was reused**, so this is a clean-source check, not a new cold-install or operating-system compatibility claim. The earlier fresh-environment installation record remains in [build history](BUILD_STATUS.md). Test counts, durations, report hashes and source-review scope are in the [aggregate review](release_review.json); raw JUnit reports and operator logs remain ignored locally.

### Numerical reproduction

The following existing reports matched byte for byte after rebuilding from authorized private saved evidence:

| Published report | Reproduction |
| --- | --- |
| [Weak-validation evaluation](epoch8_weak_validation.json) | Exact; macro AUC 0.847995649271 |
| [Expert-development evaluation](epoch8_expert_development.json) | Exact; macro AUC 0.853143219751 |
| [Training history](experiment_history.json) | Exact; selection remains epoch 8 |
| [Error analysis](error_analysis.json) | Exact; cutoff remains illustrative 0.5 |

The [evaluation](PER_FINDING_EVALUATION.md#reproduce-with-authorized-private-outputs), [history](TRAINING_HISTORY.md#rebuild-the-record-without-training) and [error-analysis](ERROR_ANALYSIS.md#reproduce) instructions describe the required private inputs and commands. Full training reproduction still needs the original cache and other authorized assets. This review reran aggregate reconstruction; it did not regenerate the plotted images or run the historical training notebooks.

### Publication boundaries

The reviewed Git file set contains no private weights, MRI files, report/label tables, study-level predictions or runtime folders. Representative ignore probes passed, and the archived notebook contains no execution outputs. The allowlist/credential/identifier checks are a bounded release audit, not a universal security guarantee. Existing public figures/media retain their earlier visual review; no new media were created here.

All local documentation links and Markdown heading anchors were checked. External links were inventoried but not exhaustively live-tested; access to Kaggle versions/data may require the appropriate account and authorization. Loopback app links require a running local server. No paid service, GPU, training, new competition submission or public inference deployment was used for this task.

## Remaining work outside this timeline

The 26 October–2 November documentation/evidence timeline is complete. The pooling comparison remains deferred until complete authorized cache inputs, tested CPU implementations and representative profiling are available. Broader scanner/site validation, calibration, real compressed-study replay and clinical use are not established by this release. The reused 58-study expert development set remains a development diagnostic.

The separate Raptor/S6 reproduction is not the deployed model. The [model card](MODEL_CARD.md), [credits](../THIRD_PARTY.md) and [portfolio wording](PORTFOLIO.md) preserve the same result, data and attribution boundaries.
