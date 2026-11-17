"""Paired person-cluster percentile bootstrap with task-specific metrics."""
from __future__ import annotations
from typing import Sequence
import numpy as np

def paired_qid_bootstrap(
    base_outcomes: Sequence[dict[str, object]],
    finetuned_outcomes: Sequence[dict[str, object]],
    *,
    metric_names,
    compute_metrics,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Paired percentile intervals while keeping birth/death rows together."""
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    base_by_case = {str(row["case_id"]): row for row in base_outcomes}
    ft_by_case = {str(row["case_id"]): row for row in finetuned_outcomes}
    if set(base_by_case) != set(ft_by_case):
        raise ValueError("paired bootstrap requires identical evaluation cases")
    qids = sorted({str(row["qid"]) for row in base_outcomes})
    base_by_qid = {
        qid: [row for row in base_outcomes if row["qid"] == qid]
        for qid in qids
    }
    ft_by_qid = {
        qid: [row for row in finetuned_outcomes if row["qid"] == qid]
        for qid in qids
    }
    point_base = compute_metrics(base_outcomes)
    point_ft = compute_metrics(finetuned_outcomes)
    differences = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(num_resamples):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        base_sample = [row for qid in sampled for row in base_by_qid[str(qid)]]
        ft_sample = [row for qid in sampled for row in ft_by_qid[str(qid)]]
        base_metrics = compute_metrics(base_sample)
        ft_metrics = compute_metrics(ft_sample)
        for name in metric_names:
            base_value = base_metrics[name]
            ft_value = ft_metrics[name]
            if base_value is not None and ft_value is not None:
                differences[name].append(float(ft_value) - float(base_value))
    alpha = 1.0 - confidence_level
    intervals = {}
    for name, values in differences.items():
        base_value = point_base[name]
        ft_value = point_ft[name]
        if base_value is None or ft_value is None or not values:
            continue
        lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        estimate = float(ft_value) - float(base_value)
        intervals[name] = {
            "base": float(base_value),
            "finetuned": float(ft_value),
            "difference": {
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "excludes_zero": bool(lower > 0 or upper < 0),
            },
        }
    return {
        "method": "paired_qid_cluster_percentile",
        "difference_definition": "finetuned_minus_baseline",
        "confidence_level": confidence_level,
        "num_resamples": num_resamples,
        "num_qid_clusters": len(qids),
        "seed": seed,
        "metrics": intervals,
    }

