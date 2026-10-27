# Controlled pooling experiment: design only

This is the **30 October timeline task**, completed early on **27 October 2026**. The protocol is written and checked. **The experiment has not run; results are null and execution is disabled.** There is no trained mean-pooling candidate, measured CPU training time or new score.

The question is whether averaging valid MRI windows changes validation ranking compared with the existing finding-specific attention, while keeping the rest of training matched. The [error analysis](ERROR_ANALYSIS.md) identifies MCL and lateral meniscus as useful predeclared diagnostics, but it does not establish that pooling caused their errors or that mean pooling will improve them.

The machine-readable specification is [pooling_protocol.json](pooling_protocol.json). Its [design-check report](pooling_protocol_check.json) verifies source bindings and shared-setting constraints. A valid protocol does not imply that data are ready, the runner is implemented or training fits the compute limit.

## The two arms

| Component | Attention control | Mean candidate |
| --- | --- | --- |
| Encoder | Trainable ConvNeXt-Tiny | Same encoder and initial tensors |
| Window token | Shared 768→256 projection, LayerNorm, SiLU, dropout 0.2, plus slot embedding | Same structure and initial tensors |
| Pooling | Learned, finding-specific gated attention over valid windows | Uniform mean over all valid windows |
| Finding logits | Twelve finding weight vectors and biases | Same structure and initial tensors |
| Initialization | Original pretraining `initial_model.pt` | Same shared tensors from that file |

For a valid token `z_i`, attention assigns weights `a_ci` separately for finding `c`, then computes `v_c = sum_i(a_ci * z_i)`. The finding logit is `w_c · v_c + b_c`. Mean pooling instead uses `v = sum_valid(z_i) / number_valid`, then `w_c · v + b_c` for each finding. It averages windows globally, rather than averaging slot means. Invalid windows have zero pooling weight; a study with no valid windows must fail.

The candidate removes the `av`, `au` and `attention` layers. From the archived layer shapes, the control has **28,089,208 parameters**, while the proposed candidate has **28,021,868**, a difference of **67,340**. These are source-derived counts; the candidate has not been constructed. The comparison measures the pooling method together with its attention-module capacity. It cannot isolate pooling independently of parameter count.

Both arms need to be trained. Replacing pooling in the already trained epoch-8 model would evaluate a head/encoder combination optimized for a different operation. Likewise, the historical 0.833 public score is context, not the new control arm's result. The CPU control must be rerun under the same conditions as the candidate.

## Fixed data and learning settings

| Setting | Both arms |
| --- | --- |
| Data | Original 3655 fit / 637 weak-validation / 58 expert-development assignments, with unchanged group boundaries |
| Labels | Original Pilkwang YES/NO soft targets; UNK masked; original explicit binary labels for AUC |
| Data identities | Exact checkpoint-bound label table, pixel index, pixel key and both cache-manifest hashes |
| Inputs | Existing 224×224 cached triplets, 140 mm crop, five slots, unchanged normalization and missing-window masks |
| Sampling | Two windows per slot during training; all eight per slot during validation, 40 before masking |
| Initialization | Original pretraining-state SHA-256 `bf6c454ddb7f1ed055f4fbeaca51a46b655444d6a40108e725a7795f4ab17da9` |
| Seed | One paired seed: 42 |
| Batch / accumulation | 2 / 4; same ordered studies, sampled window references and intensity-gain draws |
| Optimizer | Fresh AdamW per arm; encoder LR 3e-5, head LR 3e-4, weight decay 1e-4, betas 0.9/0.999, epsilon 1e-8; gradient clip 5 |
| Schedule | 40-update warmup followed by cosine decay to a 0.1 multiplier floor |
| Exposure | Eight complete epochs per arm; early stopping disabled for matched exposure |
| Loss | Original masked soft BCE, normalized over known entries across each accumulation group |
| Platform | Same local CPU, float32, autocast/scaler disabled, two compute threads, loader workers 0 |

CPU precision, worker settings and fixed-epoch exposure are shared overrides relative to the historical GPU run. They are not differences between the new arms. Full source, environment, thread settings and input hashes must be recorded for any future run; the preparation environment in the old configuration is not a complete GPU-training lockfile.

Copy the encoder, projection, slot embedding, finding weights/biases and normalization buffers from the same initial state and compare every shared tensor before training. Only the attention-specific keys may be absent in the mean model. Reset Python, NumPy and PyTorch training RNGs after construction/loading, so different constructor RNG consumption cannot change dropout or augmentation draws. Preserve the original epoch-based order/window sampler and privately record matching hashes of study order, sampled references and augmentation draws for each epoch.

There are `ceil(ceil(3655 / 2) / 4) = 457` accumulation groups per epoch, or **3656 planned optimizer updates per arm**. The historical run had 3655 successful updates because AMP skipped one group. This proposed float32 comparison instead aborts on nonfinite loss/gradients and reports an incomplete pair; it does not silently skip different updates in the two arms.

Before any fitting, the future implementation must pass CPU tests for masked-window perturbation invariance, window permutation invariance with slot metadata preserved, rejection of empty bags, finite gradients, exact shared initial tensors and unchanged input/loss behavior. A one-valid-window toy case should give the same pooled token for both methods. Those model tests and the CPU-capable runner are requirements, not completed work in this design task. The archived training driver is GPU-specific and remains unchanged.

## Checkpoint selection and comparison

After each complete 637-study validation, each arm independently retains a checkpoint only if masked soft BCE improves by more than **0.0001**, matching the original selection rule. Both arms finish all eight epochs. Select their checkpoints using that rule before comparing methods.

The primary endpoint is **weak-validation macro ROC AUC at each arm's BCE-selected checkpoint**, calculated across the same 12 targets and exact study IDs with the original known-label masks. Report all per-finding AUCs and known/positive/negative counts, BCE, ranking errors, elapsed time and peak process RSS. MCL and lateral meniscus are predeclared secondary diagnostics, not targets chosen after inspecting the new results.

The predeclared pilot screen calls mean pooling *promising* only if its macro AUC exceeds the paired control by at least **0.005** and its weak-validation soft BCE is no worse. That margin is a practical screening choice, not a statistical significance threshold or power calculation. Report the full differences even when the screen fails. An AUC/BCE tradeoff, smaller difference, implementation failure or incomplete pair does not establish an improvement. One paired seed cannot establish robustness across seeds or independent clinical performance.

Expert58 may be described once after both selections are locked, with its repeated reuse stated clearly. It cannot choose a pooling method, threshold or blend. Leaderboard scores and final competition selections are outside this experiment. The existing 0.5 cutoff remains an illustrative diagnostic; no operating threshold or calibration is fitted.

## CPU feasibility and budget gate

The proposed future ceiling is **120 total wall-clock minutes**, including profiling, both sequential arms, validation and saving. It allows one seed pair, two compute threads, zero loader workers, **4 GiB peak training-process RSS**, zero GPU runner-hours and zero paid services. This is a conservative proposed limit, not a benchmark or compute already used. The checker cannot start a run.

Once verified inputs and a tested CPU implementation exist, a disposable profile may use three warmup and five timed full accumulation groups per arm, plus one warmup and three timed validation batches with the full 40-window input contract. Use fitting studies for optimization probes; discard probe weights/optimizer state and reset both arms before fitting. The complete profiling stage has a **five-minute hard ceiling**, included in the total limit. Failure to complete representative profiles means defer, rather than extrapolate from smaller inputs.

Use median timed observations, count startup/warmup excess separately, and estimate the complete eight-epoch pair, including 457 training groups and 319 validation batches per epoch per arm. Add a **1.5 safety factor** to the projected work and a **five-minute saving reserve**. The formula and required inputs are in the protocol JSON. Cold GPU timings from the old run cannot establish CPU feasibility. The safety factor is a planning margin, not a guarantee; the wall/RSS guards remain mandatory during any run.

If either arm cannot finish under the ceilings, preserve the failure/feasibility evidence and mark the comparison incomplete. Do not shorten one arm, reduce its studies/windows, or compare a partial epoch with a complete model. A frozen-feature head study would answer a different question and would need its own protocol. There are no automatic retries, resumes or budget extensions.

Current project readiness is **deferred**. The [31 October task's feasibility audit](POOLING_FEASIBILITY.md), performed on 27 October, found no matching image shards in the audited project directories or seven saved result archives; the three replay studies are all weak validation. It stopped before benchmarking and records no accuracy results:

| Requirement | Current evidence |
| --- | --- |
| Label/split table | Exact historical hash verified during evaluation recovery |
| Pretraining initial state and pixel index | Already downloaded preparation archive contains both; streamed SHA-256 checks match the checkpoint configuration, without deserializing weights |
| Full 4350-study pixel caches | No verified complete cache mapped into the local project; the three replay cases are not a fitting dataset |
| Mean model and CPU runner | Not implemented in this design-only task |
| Paired CPU training time and peak RSS | Not measured |
| Training/accuracy results | None |

The original initialization member is 112,429,834 bytes and the pixel index is 1,274,599 bytes in the saved preparation archive. Their presence does not substitute for the image shards. The checker verifies the public design/source contract without loading these private files. Data access, implementation correctness and resource feasibility still need separate checks.

## Artifacts and verification

If a future pair is feasible and explicitly enabled under the applicable constraints, retain per-arm configuration, source/environment manifest, shared-initial-state checks, input-draw hashes, full training history, selected/last checkpoint identities, paired validation predictions, per-finding metrics, resource profiles and the comparison outcome. Keep weights, study/group IDs, individual predictions, images and reports private; publish only reviewed aggregate evidence. An incomplete run must not acquire a fabricated result row.

For today's design check:

```powershell
python -S scripts/verify_pooling_protocol.py --output .runtime/pooling_protocol_check.json
python -m pytest -q tests/test_pooling_protocol.py
```

The first command must report `POOLING_PROTOCOL_VALID` and `execution_status: DEFERRED`, with `results: null`. Its output should match `docs/pooling_protocol_check.json`. The standard-library checker binds the original configuration/model/reader/trainer sources, rejects per-arm training overrides, trained-checkpoint initialization, changed shared settings, expert-based selection, enabled/GPU execution and invented results/readiness. Its tests use only public source files; no model is constructed or trained.

The feasibility decision is now recorded in [POOLING_FEASIBILITY.md](POOLING_FEASIBILITY.md). Execution remains deferred under the current CPU-only/free-service constraints. The deployed epoch-8 model and all existing results are unchanged; this protocol provides no evidence of a new score.
