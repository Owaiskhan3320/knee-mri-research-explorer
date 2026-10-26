"""CPU-only, descriptive audit of public report labels.

This module never changes training labels, chooses a source, calibrates a
threshold, reads the JEV gold74 file, or exports report text.  The comparison
uses the caller's fixed patient/report-group split.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


UID = "StudyInstanceUID"
SPLITS = ("fit", "weak_validation", "gold")
SOURCE_URLS = {
    "pilkwang": "https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels",
    "steven_v1": "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels",
    "steven_v2": "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels",
    "steven_v4": "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels",
    "jev_hybrid": "https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels",
}
CALIBRATION_CAVEAT = (
    "Development diagnostics only. Steven v2 used the 58 expert studies to "
    "investigate the Synovitis-from-Effusion fill, and v4 was compared on them. "
    "JEV HYBRID selected its source per target using those 58 studies; its "
    "text-ensemble weights and blend were developed using a larger labelled "
    "set containing them. Its 74-study set includes 16 manually report-read "
    "cases, not 74 independent image-expert annotations. Consequently gold "
    "results here cannot independently rank these sources. No labels, weights, "
    "thresholds or priors are selected using these metrics."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def _as_frames(frames, targets):
    if isinstance(frames, Mapping):
        parts = []
        for split in SPLITS:
            if split in frames:
                part = frames[split].copy(deep=True)
                if "split" in part:
                    assert part["split"].eq(split).all(), f"Wrong split in {split} frame"
                part["split"] = split
                parts.append(part)
        assert parts, "No recognised split frames supplied"
        df = pd.concat(parts, ignore_index=True)
    else:
        df = frames.copy(deep=True)
    if UID not in df and "StudyUID" in df:
        df = df.rename(columns={"StudyUID": UID})
    required = [UID, "split", "verified_group", *targets]
    assert set(required) <= set(df), f"Missing frame columns: {set(required) - set(df)}"
    assert df[UID].notna().all(), "Missing frame UID"
    df[UID] = df[UID].astype(str)
    assert df[UID].str.len().gt(0).all() and df[UID].is_unique, "Duplicate/empty frame UID"
    assert df["split"].isin(SPLITS).all(), "Unexpected split name"
    assert df["verified_group"].notna().all(), "Missing verified_group"
    assert df["verified_group"].astype(str).str.len().gt(0).all(), "Empty verified_group"
    assert df.groupby("verified_group")["split"].nunique().le(1).all(), "Group crosses splits"
    return df.sort_values(["split", UID]).reset_index(drop=True)


def _probability_array(df, targets, label):
    assert set(targets) <= set(df), f"{label}: missing target columns {set(targets) - set(df)}"
    original = df[targets]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    bad_strings = original.notna() & numeric.isna()
    assert not bad_strings.any().any(), f"{label}: nonnumeric probability"
    values = numeric.to_numpy(dtype=np.float64)
    invalid = ~np.isnan(values) & (~np.isfinite(values) | (values < 0) | (values > 1))
    assert not invalid.any(), f"{label}: probability outside [0, 1] or infinite"
    return values


def _read_source(path, name, targets):
    frame = pd.read_csv(path, dtype={UID: str})
    assert UID in frame and frame[UID].notna().all(), f"{name}: UID missing"
    assert frame[UID].str.len().gt(0).all() and frame[UID].is_unique, f"{name}: duplicate/empty UID"
    _probability_array(frame, targets, name)
    return frame.set_index(UID, drop=False)


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(y) and np.unique(y).size == 2 else None


def _correlation(a, b, rank=False):
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    if rank:
        a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def _frame_hash(frame, targets):
    columns = [UID, "split", "verified_group", *targets]
    columns += [f"{t}__binary" for t in targets if f"{t}__binary" in frame]
    payload = frame[columns].sort_values(UID).to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(payload.encode()).hexdigest()


def run_label_audit(pilkwang_path, steven_root, jev_root, frames, out_dir, targets):
    """Write descriptive audits and return a JSON-serialisable manifest.

    ``frames`` is either a DataFrame or a mapping from ``fit``,
    ``weak_validation`` and ``gold`` to DataFrames. Each row needs
    StudyInstanceUID, verified_group, target soft values and, when available,
    target+'__binary'. Gold metrics use __binary or exact 0/1 gold targets.

    Pilkwang is required. Each Steven file and the JEV score file are optional:
    a missing optional file is recorded rather than silently substituted.
    Missing probabilities remain missing. The function reads no report text.
    """
    targets = list(targets)
    assert targets and len(set(targets)) == len(targets), "Invalid target order"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _as_frames(frames, targets)
    initial_hash = _frame_hash(df, targets)
    uids = df[UID].tolist()
    split_array = df["split"].to_numpy()
    n, k = len(df), len(targets)
    required_path = Path(pilkwang_path)
    assert required_path.is_file(), f"Required Pilkwang file missing: {required_path}"
    candidates = [("pilkwang", required_path, True)]
    for name, filename in (
        ("steven_v1", "llm_labels_full.csv"),
        ("steven_v2", "llm_labels_v2.csv"),
        ("steven_v4", "llm_labels_v4_blend.csv"),
    ):
        candidates.append((name, Path(steven_root) / filename if steven_root else None, False))
    candidates.append(("jev_hybrid", Path(jev_root) / "HYBRID_labels_scores.csv" if jev_root else None, False))

    sources, hashes, missing = {}, {}, []
    for name, path, required in candidates:
        if path is None or not path.is_file():
            missing.append({"source": name, "path": str(path) if path else None, "required": required})
            continue
        raw = _read_source(path, name, targets)
        aligned = raw.reindex(uids)
        probabilities = _probability_array(aligned, targets, name)
        finite = np.isfinite(probabilities)
        if name == "pilkwang":
            verdict_columns = [f"{t}__verdict" for t in targets]
            assert set(verdict_columns) <= set(raw), "Pilkwang verdict columns missing"
            verdict = aligned[verdict_columns].apply(lambda x: x.astype("string").str.strip().str.upper())
            unexpected = verdict.notna() & ~verdict.isin(["YES", "NO", "UNK"])
            assert not unexpected.any().any(), "Unexpected Pilkwang verdict"
            known = verdict.isin(["YES", "NO"]).to_numpy(dtype=bool) & finite
            unknown = verdict.eq("UNK").fillna(False).to_numpy(dtype=bool)
            assert not (verdict.isin(["YES", "NO"]).to_numpy(dtype=bool) & ~finite).any(), "Known Pilkwang target has no probability"
            policy = "Finite score AND verdict YES/NO; UNK is masked regardless of numeric score."
        elif name in ("steven_v1", "steven_v2"):
            known = finite & ~np.isclose(probabilities, 0.5, atol=1e-12, rtol=0)
            unknown = finite & ~known
            policy = "Exact 0.5 is unaddressed under the author's v1/v2 convention. v2 Synovitis includes derived fills."
        elif name == "steven_v4":
            unknown = finite & np.isclose(probabilities, 0.25, atol=1e-12, rtol=0)
            known = finite & ~unknown
            policy = "PROVISIONAL: exact 0.25 masked following downstream discussion 746792; dataset card only explicitly documents v1 0.5. Midpoint 0.5 is separately flagged, not automatically missing."
        else:
            known, unknown = finite.copy(), np.zeros((n, k), dtype=bool)
            policy = "All finite HYBRID probabilities are available; no numeric unknown sentinel is documented. Source-derived unknown flags are separate."
        item = {
            "path": str(path), "raw": raw, "p": probabilities, "finite": finite,
            "known": known, "unknown": unknown, "policy": policy,
        }
        sources[name] = item
        hashes[name] = {
            "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
            "rows": len(raw), "columns": list(raw.columns),
            "matched_frame_uids": int(pd.Index(uids).isin(raw.index).sum()),
            "source_uids_outside_fixed_frames": int((~raw.index.isin(uids)).sum()),
            "url": SOURCE_URLS[name], "known_policy": policy,
        }

    # Audit the persisted supervision without changing it. Scores and masks must
    # reproduce Pilkwang on the report-labelled fit/validation rows.
    pil = sources["pilkwang"]
    base = _probability_array(df, targets, "fixed frames")
    weak_rows = np.isin(split_array, ["fit", "weak_validation"])
    assert np.array_equal(np.isfinite(base[weak_rows]), pil["known"][weak_rows]), "Fixed weak-label mask differs from Pilkwang"
    assert np.allclose(base[weak_rows][pil["known"][weak_rows]], pil["p"][weak_rows][pil["known"][weak_rows]], atol=1e-7, rtol=0), "Fixed weak-label values differ from Pilkwang"

    # JEV blending moves .25 and .28, so do not infer missingness by comparing
    # its final values to those constants. Reconstruct provenance flags only.
    if "jev_hybrid" in sources:
        source_known = np.zeros((n, k), dtype=bool)
        available = np.zeros((n, k), dtype=bool)
        for j, target in enumerate(targets):
            origin = "pilkwang" if target in ("ACL", "MCL", "Fracture") else "steven_v4"
            if origin in sources:
                source_known[:, j] = sources[origin]["known"][:, j]
                available[:, j] = sources[origin]["finite"][:, j]
        sources["jev_hybrid"]["hard_swap_source_known"] = source_known
        sources["jev_hybrid"]["hard_swap_source_available"] = available

    coverage, sentinels = [], []
    for name, item in sources.items():
        for split in (*SPLITS, "all"):
            row_mask = split_array == split if split != "all" else np.ones(n, dtype=bool)
            for j, target in enumerate(targets):
                p, finite, known = item["p"][row_mask, j], item["finite"][row_mask, j], item["known"][row_mask, j]
                q = p[finite]
                coverage.append({
                    "source": name, "source_path": item["path"], "split": split, "target": target,
                    "studies": int(row_mask.sum()), "finite_scores": int(finite.sum()),
                    "known_by_stated_policy": int(known.sum()), "unknown_by_stated_policy": int(item["unknown"][row_mask, j].sum()),
                    "no_score": int((~finite).sum()),
                    "known_above_half": int((known & (p > 0.5)).sum()),
                    "known_below_half": int((known & (p < 0.5)).sum()),
                    "known_at_half": int((known & np.isclose(p, 0.5, atol=1e-12, rtol=0)).sum()),
                    "mean_probability_finite": float(q.mean()) if len(q) else None,
                    "known_policy": item["policy"],
                    "hard_swap_source_unknown": int((item["hard_swap_source_available"][row_mask, j] & ~item["hard_swap_source_known"][row_mask, j]).sum()) if "hard_swap_source_known" in item else None,
                })
                for value in (0.0, 0.08, 0.25, 0.28, 0.5, 0.68, 0.82, 0.94, 1.0):
                    equal = finite & np.isclose(p, value, atol=1e-12, rtol=0)
                    sentinels.append({
                        "source": name, "source_path": item["path"], "split": split, "target": target,
                        "value": value, "count": int(equal.sum()),
                        "count_known_by_stated_policy": int((equal & known).sum()),
                        "interpretation": "Numeric mass only; source-specific missingness policies are in provenance.json.",
                    })

    pair_rows, conflict_rows = [], []
    for (name_a, a), (name_b, b) in itertools.combinations(sources.items(), 2):
        for split in (*SPLITS, "all"):
            rows = split_array == split if split != "all" else np.ones(n, dtype=bool)
            for j, target in enumerate(targets):
                both = rows & a["finite"][:, j] & b["finite"][:, j]
                common = rows & a["known"][:, j] & b["known"][:, j]
                for policy, mask in (("all_finite", both), ("both_known_by_stated_policy", common)):
                    pa, pb = a["p"][mask, j], b["p"][mask, j]
                    delta = np.abs(pa - pb)
                    directional = (pa != 0.5) & (pb != 0.5)
                    pair_rows.append({
                        "source_a": name_a, "source_b": name_b,
                        "source_a_path": a["path"], "source_b_path": b["path"],
                        "split": split, "target": target, "comparison_policy": policy,
                        "n": int(mask.sum()), "mean_abs_difference": float(delta.mean()) if len(delta) else None,
                        "median_abs_difference": float(np.median(delta)) if len(delta) else None,
                        "p95_abs_difference": float(np.quantile(delta, 0.95)) if len(delta) else None,
                        "max_abs_difference": float(delta.max()) if len(delta) else None,
                        "pearson": _correlation(pa, pb), "spearman": _correlation(pa, pb, rank=True),
                        "direction_comparable": int(directional.sum()),
                        "opposite_sides_of_half": int((directional & ((pa > 0.5) != (pb > 0.5))).sum()),
                        "strong_conflicts_0p2_vs_0p8": int((((pa <= 0.2) & (pb >= 0.8)) | ((pb <= 0.2) & (pa >= 0.8))).sum()),
                    })
                # One row per actual study/target/pair; no duplicated "all" rows.
                if split != "all":
                    pa, pb = a["p"][:, j], b["p"][:, j]
                    opposite = common & (pa != 0.5) & (pb != 0.5) & ((pa > 0.5) != (pb > 0.5))
                    for i in np.flatnonzero(opposite):
                        conflict_rows.append({
                            UID: uids[i], "split": split, "target": target,
                            "source_a": name_a, "source_b": name_b,
                            "source_a_path": a["path"], "source_b_path": b["path"],
                            "probability_a": float(pa[i]), "probability_b": float(pb[i]),
                            "abs_difference": float(abs(pa[i] - pb[i])),
                            "strong_conflict_0p2_vs_0p8": bool(min(pa[i], pb[i]) <= 0.2 and max(pa[i], pb[i]) >= 0.8),
                            "interpretation": "Report-label disagreement, not an adjudicated error.",
                        })

    # Record how the Steven versions alter missingness and values; no ranking.
    transitions = []
    if "steven_v1" in sources:
        first = sources["steven_v1"]
        for later_name in ("steven_v2", "steven_v4"):
            if later_name not in sources:
                continue
            later = sources[later_name]
            for split in SPLITS:
                rows = split_array == split
                for j, target in enumerate(targets):
                    common = rows & first["finite"][:, j] & later["finite"][:, j]
                    changed = common & ~np.isclose(first["p"][:, j], later["p"][:, j], atol=1e-12, rtol=0)
                    transitions.append({
                        "from_source": "steven_v1", "to_source": later_name, "split": split, "target": target,
                        "common_finite": int(common.sum()), "changed_values": int(changed.sum()),
                        "v1_unaddressed_to_later_policy_known": int((common & ~first["known"][:, j] & later["known"][:, j]).sum()),
                        "v1_addressed_to_later_policy_unknown": int((common & first["known"][:, j] & ~later["known"][:, j]).sum()),
                        "interpretation": "Derived fill/blend changes; not new independent annotations.",
                    })

    gold_rows = split_array == "gold"
    gold_metrics = []
    for j, target in enumerate(targets):
        col = f"{target}__binary"
        truth = pd.to_numeric(df[col] if col in df else df[target], errors="coerce").to_numpy(dtype=float)
        valid_truth = np.isfinite(truth) & np.isin(truth, [0.0, 1.0]) & gold_rows
        for name, item in sources.items():
            for policy, score_mask in (("all_finite", item["finite"][:, j]), ("known_by_stated_policy", item["known"][:, j])):
                mask = valid_truth & score_mask
                y, p = truth[mask], item["p"][mask, j]
                clipped = np.clip(p, 1e-7, 1 - 1e-7)
                gold_metrics.append({
                    "source": name, "source_path": item["path"], "target": target,
                    "scope": "gold_development_only", "score_policy": policy,
                    "n": int(mask.sum()), "positives": int(y.sum()), "auc": _auc(y, p),
                    "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
                    "binary_cross_entropy": float(np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))) if len(y) else None,
                    "truth_column": col if col in df else target,
                    "calibration_caveat": CALIBRATION_CAVEAT,
                })

    gold_table = pd.DataFrame(gold_metrics)
    gold_macro = []
    if len(gold_table):
        for (source, policy), table in gold_table.groupby(["source", "score_policy"], sort=False):
            gold_macro.append({
                "source": source, "source_path": sources[source]["path"],
                "scope": "gold_development_only", "score_policy": policy,
                "targets_with_auc": int(table["auc"].notna().sum()),
                "macro_auc": float(table["auc"].mean()) if table["auc"].notna().any() else None,
                "interpretation": "Mean over available targets only; compare counts and coverage, not just this mean.",
                "calibration_caveat": CALIBRATION_CAVEAT,
            })
    artifacts = {
        "coverage_by_target_split.csv": pd.DataFrame(coverage),
        "unknown_sentinel_counts.csv": pd.DataFrame(sentinels),
        "pairwise_label_comparisons.csv": pd.DataFrame(pair_rows),
        "label_conflicts.csv": pd.DataFrame(conflict_rows, columns=[UID, "split", "target", "source_a", "source_b", "source_a_path", "source_b_path", "probability_a", "probability_b", "abs_difference", "strong_conflict_0p2_vs_0p8", "interpretation"]),
        "steven_version_transitions.csv": pd.DataFrame(transitions, columns=["from_source", "to_source", "split", "target", "common_finite", "changed_values", "v1_unaddressed_to_later_policy_known", "v1_addressed_to_later_policy_unknown", "interpretation"]),
        "gold_development_only_metrics.csv": gold_table,
        "gold_development_only_macro.csv": pd.DataFrame(gold_macro),
    }
    for filename, table in artifacts.items():
        table.to_csv(out / filename, index=False)

    assert _frame_hash(df, targets) == initial_hash, "Audit changed fixed supervision"
    assumptions = {
        "reviewed_on": "2026-10-18",
        "steven_dataset_version_seen_in_public_ui": 6,
        "jev_dataset_version_seen_in_public_ui": 5,
        "actual_mounted_files": "SHA-256 values below identify actual input bytes; public UI version numbers are context, not a substitute for hashes.",
        "pilkwang": "Use score as soft target only where __verdict is YES/NO; __conf is not a class target. No change to existing supervision.",
        "steven_v1_v2": "Author explicitly documents 0.5 as report-not-addressed. v2 derives undecided Synovitis from Effusion; it is not all independent extraction.",
        "steven_v4": "Probability blend of v2 and lixin73 GPT-5.6-Sol. Exact 0.25 is provisionally flagged unaddressed following a downstream user's detailed recipe; the dataset card's generic 0.5 statement is not applied blindly to v4. No confidence weights are derived from v4 in this audit.",
        "steven_v4_sentinel_reference": "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792",
        "jev_hybrid": "0.95 hard-swap + 0.05 JEV ensemble. Hard-swap uses Pilkwang ACL/MCL/Fracture and Steven v4 for other targets. No fixed final-value missing sentinel assumed. The origin-known flag does not adjudicate JEV's own contribution.",
        "jev_confidence": "FINAL2 _agree and _tier fields measure six JEV text members, not confidence of HYBRID. They are not loaded or used as training weights.",
        "licenses_seen_on_public_cards": {"steven": "CC0-1.0", "jev": "CC0-1.0"},
        "eligibility": "Public report-derived label assets are being audited, not controlled-access external MRI data. A dataset license does not override competition rules or the competition data terms.",
        "gold_caveat": CALIBRATION_CAVEAT,
        "deliberately_not_read": ["gold74_eval_set.csv", "HYBRID_labels_binary.csv", "HYBRID_thresholds.json", "FINAL2_labels_with_confidence.csv"],
        "pairwise_comparisons": "Both all-finite and policy-known intersections are reported. Opposite sides of 0.5 and <=0.2 versus >=0.8 conflicts are descriptive fixed rules, not optimised thresholds or adjudicated errors.",
        "no_raw_reports": True, "no_training_label_selection": True,
    }
    manifest = {
        "complete": True, "optional_sources_complete": len(missing) == 0,
        "target_order": targets, "frame_rows": n,
        "split_counts": {s: int((split_array == s).sum()) for s in SPLITS},
        "fixed_frame_semantic_sha256": initial_hash,
        "training_supervision_unchanged": "Pilkwang soft scores with YES/NO mask; equality asserted against supplied fit/weak-validation frames.",
        "input_files": hashes, "missing_optional_files": missing,
        "source_count": len(sources), "card_assumptions": assumptions,
        "output_files": {name: {"path": str(out / name), "sha256": _sha256(out / name), "rows": len(table)} for name, table in artifacts.items()},
        "gpu_used": False,
    }
    _json_write(out / "provenance.json", manifest)
    _json_write(out / "card_assumptions.json", assumptions)
    _json_write(out / "audit_status.json", {"complete": True, "optional_sources_complete": len(missing) == 0, "gpu_used": False, "training_labels_changed": False})
    print("LABEL_AUDIT_COMPLETE", json.dumps({"sources": list(sources), "rows": n, "missing_optional": len(missing), "gpu_used": False}))
    return manifest
