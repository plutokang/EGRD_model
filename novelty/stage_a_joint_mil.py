from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

def import_local_module(module_name: str, file_name: str) -> Any:
    path = Path(__file__).resolve().parent / file_name
    if not path.is_file():
        raise ModuleNotFoundError(f"Cannot find {module_name} at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


joint_text = import_local_module("stage_a_joint_base", "stage_a_joint_base.py")
joint_score_text = import_local_module("stage_a_joint_score", "stage_a_joint_score.py")

JOINT_LABELS = joint_text.JOINT_LABELS
JOINT_TO_ID = joint_text.JOINT_TO_ID
COVERED_INTERNAL_LABELS = JOINT_LABELS[1:]
TextLookup = joint_text.TextLookup
build_text_lookup = joint_text.build_text_lookup
class_weights = joint_text.class_weights
device_from_arg = joint_text.device_from_arg
load_novelty_index = joint_text.load_novelty_index
load_ranking_splits = joint_text.load_ranking_splits
metadata_line = joint_text.metadata_line
ordinal_class_probs = joint_text.ordinal_class_probs
ordinal_pos_weights = joint_text.ordinal_pos_weights
ordinal_targets = joint_text.ordinal_targets
ranking_priors = joint_text.ranking_priors
safe_float = joint_text.safe_float
sample_train_rows = joint_text.sample_train_rows
set_seed = joint_text.set_seed
write_jsonl = joint_text.write_jsonl

label_metrics = joint_score_text.label_metrics
parse_thresholds = joint_score_text.parse_thresholds
score_metrics = joint_score_text.score_metrics
score_to_label_ids = joint_score_text.score_to_label_ids


PRED_PAIR_LABEL_KEYS = ["calibrated_pair_label", "pred_pair_label", "raw_pred_pair_label"]
PRED_PAIR_SCORE_KEYS = ["calibrated_score", "pred_coverage_score", "rank_score"]
GOLD_PAIR_LABEL_KEYS = ["gold_pair_label", "coverage_label"]
GOLD_PAIR_SCORE_KEYS = ["gold_coverage_score", "coverage_score"]
DEFAULT_CLASS_SCORE_MEANS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
GRAPH_RELATIONS = [
    "extends",
    "overlaps_with",
    "different_from",
    "shares_technique",
    "applies_to_new_context",
    "combines",
]
GRAPH_DIRECTIONS = ["out", "in"]
GRAPH_RELATION_TOKENS = [f"{direction}_{relation}" for direction in GRAPH_DIRECTIONS for relation in GRAPH_RELATIONS]
GRAPH_PATH_TOKENS = [f"{left}>{right}" for left in GRAPH_RELATION_TOKENS for right in GRAPH_RELATION_TOKENS]
GRAPH_FACET_FIELDS = [
    "primary_aspect",
    "contribution_type",
    "graph_form",
    "task_family",
    "mechanism_family",
]
GRAPH_CONSTRUCTIVE_RELATIONS = {
    "extends",
    "applies_to_new_context",
    "combines",
    "shares_technique",
}
GRAPH_CONTRAST_RELATIONS = {"different_from"}
RELATION_PRIORITY = {
    "extends": 6,
    "applies_to_new_context": 5,
    "combines": 4,
    "shares_technique": 3,
    "overlaps_with": 2,
    "different_from": 1,
}


def write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def scaled_count(value: int | float, scale: float = 20.0) -> float:
    return min(1.0, math.log1p(max(0.0, float(value))) / math.log1p(scale))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / denominator))


def normalized_entropy(counts: Counter[str] | dict[str, int | float], bucket_count: int) -> float:
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0 or bucket_count <= 1:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        prob = float(value) / total
        if prob > 0:
            entropy -= prob * math.log(prob)
    return min(1.0, entropy / math.log(float(bucket_count)))


def date_to_month_index(date: int | None) -> int | None:
    if not date:
        return None
    year = int(date) // 100
    month = int(date) % 100
    if month < 1 or month > 12:
        return None
    return year * 12 + month


def month_gap(older_date: int | None, newer_date: int | None) -> int | None:
    older = date_to_month_index(older_date)
    newer = date_to_month_index(newer_date)
    if older is None or newer is None:
        return None
    return max(0, newer - older)


def clamp_monotonic(values: list[float]) -> list[float]:
    clipped = np.asarray([min(1.0, max(0.0, float(value))) for value in values], dtype=np.float32)
    return np.maximum.accumulate(clipped).astype(float).tolist()


def class_score_means_from_dataset(dataset: "JointMilDataset") -> list[float]:
    by_label: dict[int, list[float]] = defaultdict(list)
    for item in dataset.items:
        by_label[int(item["label"])].append(float(item["score"]))
    means = []
    for idx, fallback in enumerate(DEFAULT_CLASS_SCORE_MEANS):
        values = by_label.get(idx, [])
        means.append(float(np.mean(values)) if values else float(fallback))
    return clamp_monotonic(means)


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=loss.dtype, device=loss.device)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def masked_weighted_mean(loss: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    effective = weights.to(dtype=loss.dtype, device=loss.device) * mask.to(dtype=loss.dtype, device=loss.device)
    return (loss * effective).sum() / effective.sum().clamp_min(1e-6)


def hierarchical_class_probs(
    any_logits: torch.Tensor,
    substantial_logits: torch.Tensor,
    mostly_logits: torch.Tensor,
) -> torch.Tensor:
    any_prob = torch.sigmoid(any_logits.float())
    substantial_prob = torch.sigmoid(substantial_logits.float())
    mostly_prob = torch.sigmoid(mostly_logits.float())
    p_not_covered = 1.0 - any_prob
    p_weakly_covered = any_prob * (1.0 - substantial_prob)
    p_partially_covered = any_prob * substantial_prob * (1.0 - mostly_prob)
    p_mostly_covered = any_prob * substantial_prob * mostly_prob
    probs = torch.stack(
        [
            p_not_covered,
            p_weakly_covered,
            p_partially_covered,
            p_mostly_covered,
        ],
        dim=-1,
    )
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def covered_internal_class_probs(
    any_logits: torch.Tensor,
    covered_internal_logits: torch.Tensor,
) -> torch.Tensor:
    any_prob = torch.sigmoid(any_logits.float())
    internal_probs = torch.softmax(covered_internal_logits.float(), dim=-1)
    probs = torch.cat(
        [
            (1.0 - any_prob).unsqueeze(-1),
            any_prob.unsqueeze(-1) * internal_probs,
        ],
        dim=-1,
    )
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def expected_score_from_probs(probs: torch.Tensor, class_score_means: torch.Tensor) -> torch.Tensor:
    anchors = class_score_means.to(device=probs.device, dtype=probs.dtype)
    return (probs * anchors.unsqueeze(0)).sum(dim=-1).clamp(0.0, 1.0)


def describe_scores(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.quantile(values.astype(np.float64), [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(len(values)),
        "min": round(float(values.min()), 4),
        "p10": round(float(quantiles[0]), 4),
        "p25": round(float(quantiles[1]), 4),
        "median": round(float(quantiles[2]), 4),
        "mean": round(float(values.mean()), 4),
        "p75": round(float(quantiles[3]), 4),
        "p90": round(float(quantiles[4]), 4),
        "max": round(float(values.max()), 4),
    }


def score_distribution_by_label(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        label: describe_scores(scores[labels == idx])
        for idx, label in enumerate(JOINT_LABELS)
    }


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return out


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    pos = int(y_true.sum())
    neg = int((~y_true).sum())
    if pos == 0 or neg == 0:
        return 0.0
    score_ranks = ranks(y_score.astype(np.float64))
    auc = (float(score_ranks[y_true].sum()) - pos * (pos - 1) / 2.0) / max(1.0, pos * neg)
    return round(float(auc), 4)


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-y_score.astype(np.float64), kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    rank = np.arange(1, len(sorted_true) + 1)
    precision_at_k = tp / rank
    return round(float(precision_at_k[sorted_true].sum() / pos), 4)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out: dict[str, Any] = {
        "accuracy": round(float((tp + tn) / max(1, len(y_true))), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "positive_support": int(y_true.sum()),
        "negative_support": int((~y_true).sum()),
        "pred_positive": int(y_pred.sum()),
    }
    if y_score is not None and len(y_score):
        out["auroc"] = binary_auroc(y_true.astype(np.int64), y_score)
        out["auprc"] = binary_auprc(y_true.astype(np.int64), y_score)
    return out


def enriched_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = label_metrics(y_true, y_pred)
    recalls = [float(item["recall"]) for item in metrics["per_class"].values()]
    metrics["balanced_accuracy"] = round(float(np.mean(recalls)), 4) if recalls else 0.0
    metrics["covered_binary"] = binary_metrics(y_true >= 1, y_pred >= 1)
    metrics["substantial_binary"] = binary_metrics(y_true >= 2, y_pred >= 2)
    partial = metrics["per_class"]["partially_covered"]
    mostly = metrics["per_class"]["mostly_covered"]
    weak = metrics["per_class"]["weakly_covered"]
    metrics["weak_recall"] = weak["recall"]
    metrics["weak_f1"] = weak["f1"]
    metrics["partial_precision"] = partial["precision"]
    metrics["partial_recall"] = partial["recall"]
    metrics["partial_f1"] = partial["f1"]
    metrics["mostly_f1"] = mostly["f1"]
    return metrics


def threshold_objective(metrics: dict[str, Any], objective: str) -> float:
    if objective == "macro_f1":
        return float(metrics["macro_f1"])
    if objective == "qwk":
        return float(metrics["qwk"])
    if objective == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if objective == "neg_ordinal_mae":
        return -float(metrics["ordinal_mae"])
    return (
        float(metrics["macro_f1"])
        + 0.25 * float(metrics["qwk"])
        - 0.10 * float(metrics["ordinal_mae"])
    )


def threshold_constraints_satisfied(
    metrics: dict[str, Any],
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> bool:
    partial_cap = effective_pred_ratio_cap(
        metrics,
        "partially_covered",
        max_partial_pred_ratio,
        max_partial_pred_multiplier,
    )
    mostly_cap = effective_pred_ratio_cap(
        metrics,
        "mostly_covered",
        max_mostly_pred_ratio,
        max_mostly_pred_multiplier,
    )
    return (
        float(metrics["not_covered_recall"]) >= min_not_recall
        and float(metrics["weak_recall"]) >= min_weak_recall
        and float(metrics["partial_recall"]) >= min_partial_recall
        and float(metrics["mostly_recall"]) >= min_mostly_recall
        and float(metrics["covered_binary"]["recall"]) >= min_covered_recall
        and distribution_rate(metrics, "pred_distribution", "partially_covered") <= partial_cap
        and distribution_rate(metrics, "pred_distribution", "mostly_covered") <= mostly_cap
    )


def distribution_rate(metrics: dict[str, Any], distribution_key: str, label: str) -> float:
    distribution = metrics.get(distribution_key) or {}
    total = sum(int(value) for value in distribution.values())
    if total <= 0:
        return 0.0
    return float(int(distribution.get(label, 0)) / total)


def effective_pred_ratio_cap(
    metrics: dict[str, Any],
    label: str,
    max_ratio: float,
    max_multiplier: float,
) -> float:
    caps = []
    if max_ratio > 0:
        caps.append(float(max_ratio))
    true_rate = distribution_rate(metrics, "true_distribution", label)
    if max_multiplier > 0 and true_rate > 0:
        caps.append(float(true_rate * max_multiplier))
    return min(caps) if caps else 1.0


def constrained_threshold_objective(
    metrics: dict[str, Any],
    objective: str,
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> float:
    base = threshold_objective(metrics, objective)
    partial_cap = effective_pred_ratio_cap(
        metrics,
        "partially_covered",
        max_partial_pred_ratio,
        max_partial_pred_multiplier,
    )
    mostly_cap = effective_pred_ratio_cap(
        metrics,
        "mostly_covered",
        max_mostly_pred_ratio,
        max_mostly_pred_multiplier,
    )
    partial_pred_rate = distribution_rate(metrics, "pred_distribution", "partially_covered")
    mostly_pred_rate = distribution_rate(metrics, "pred_distribution", "mostly_covered")
    deficits = [
        max(0.0, min_not_recall - float(metrics["not_covered_recall"])),
        max(0.0, min_weak_recall - float(metrics["weak_recall"])),
        max(0.0, min_partial_recall - float(metrics["partial_recall"])),
        max(0.0, min_mostly_recall - float(metrics["mostly_recall"])),
        max(0.0, min_covered_recall - float(metrics["covered_binary"]["recall"])),
        max(0.0, partial_pred_rate - partial_cap),
        max(0.0, mostly_pred_rate - mostly_cap),
    ]
    penalty = sum(deficits)
    if penalty > 0:
        return -1.0 - penalty + 0.01 * base
    return base


def constraint_summary(
    metrics: dict[str, Any],
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> dict[str, Any]:
    partial_cap = effective_pred_ratio_cap(
        metrics,
        "partially_covered",
        max_partial_pred_ratio,
        max_partial_pred_multiplier,
    )
    mostly_cap = effective_pred_ratio_cap(
        metrics,
        "mostly_covered",
        max_mostly_pred_ratio,
        max_mostly_pred_multiplier,
    )
    return {
        "min_not_covered_recall": min_not_recall,
        "min_weak_recall": min_weak_recall,
        "min_partial_recall": min_partial_recall,
        "min_mostly_recall": min_mostly_recall,
        "min_covered_recall": min_covered_recall,
        "max_partial_pred_ratio": round(float(partial_cap), 6),
        "max_mostly_pred_ratio": round(float(mostly_cap), 6),
        "actual_partial_pred_ratio": round(
            distribution_rate(metrics, "pred_distribution", "partially_covered"),
            6,
        ),
        "actual_mostly_pred_ratio": round(
            distribution_rate(metrics, "pred_distribution", "mostly_covered"),
            6,
        ),
        "satisfied": threshold_constraints_satisfied(
            metrics,
            min_not_recall,
            min_weak_recall,
            min_partial_recall,
            min_mostly_recall,
            min_covered_recall,
            max_partial_pred_ratio,
            max_mostly_pred_ratio,
            max_partial_pred_multiplier,
            max_mostly_pred_multiplier,
        ),
    }


def candidate_thresholds(values: np.ndarray, steps: int, lo: float = 0.01, hi: float = 0.99) -> list[float]:
    steps = max(4, int(steps))
    lo = min(0.99, max(0.01, float(lo)))
    hi = min(0.99, max(lo, float(hi)))
    candidates = set(float(x) for x in np.linspace(lo, hi, steps))
    if len(values):
        quantiles = np.linspace(0.02, 0.98, steps)
        candidates.update(
            float(value)
            for value in (np.quantile(values, q) for q in quantiles)
            if lo <= float(value) <= hi
        )
    ordered = sorted(round(min(hi, max(lo, value)), 6) for value in candidates)
    if len(ordered) <= steps:
        return ordered
    keep = np.linspace(0, len(ordered) - 1, steps).round().astype(np.int64)
    return [ordered[int(idx)] for idx in keep]


def include_candidate(candidates: list[float], value: float, lo: float, hi: float) -> list[float]:
    if lo <= value <= hi:
        candidates.append(float(value))
    return sorted(set(round(float(item), 6) for item in candidates))


def tune_ordered_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    objective: str,
    steps: int,
    initial_thresholds: tuple[float, float, float],
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    best_thresholds = initial_thresholds
    best_metrics = enriched_label_metrics(y_true, score_to_label_ids(score, best_thresholds))
    best_score = constrained_threshold_objective(
        best_metrics,
        objective,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    candidates = candidate_thresholds(score, steps)
    for value in initial_thresholds:
        candidates = include_candidate(candidates, value, 0.01, 0.99)
    for weak_threshold in candidates:
        for partial_threshold in candidates:
            if partial_threshold <= weak_threshold:
                continue
            for mostly_threshold in candidates:
                if mostly_threshold <= partial_threshold:
                    continue
                pred = score_to_label_ids(
                    score,
                    (
                        float(weak_threshold),
                        float(partial_threshold),
                        float(mostly_threshold),
                    ),
                )
                metrics = enriched_label_metrics(y_true, pred)
                current = constrained_threshold_objective(
                    metrics,
                    objective,
                    min_not_recall,
                    min_weak_recall,
                    min_partial_recall,
                    min_mostly_recall,
                    min_covered_recall,
                    max_partial_pred_ratio,
                    max_mostly_pred_ratio,
                    max_partial_pred_multiplier,
                    max_mostly_pred_multiplier,
                )
                if current > best_score:
                    best_score = current
                    best_thresholds = (float(weak_threshold), float(partial_threshold), float(mostly_threshold))
                    best_metrics = metrics
    best_metrics["threshold_objective"] = objective
    best_metrics["threshold_objective_score"] = round(float(best_score), 6)
    best_metrics["threshold_constraints"] = constraint_summary(
        best_metrics,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    return best_thresholds, best_metrics


def parse_threshold_range(text: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"--{name} must contain exactly 2 comma-separated values")
    lo, hi = values
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"--{name} must satisfy 0 <= lo <= hi <= 1")
    return float(lo), float(hi)


def parse_conditional_thresholds(text: str) -> tuple[float, float, float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--conditional-thresholds must contain exactly 3 comma-separated values")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("--conditional-thresholds values must be in [0, 1]")
    return float(values[0]), float(values[1]), float(values[2])


def conditional_label_ids(
    any_prob: np.ndarray,
    substantial_prob: np.ndarray,
    mostly_prob: np.ndarray,
    thresholds: tuple[float, float, float],
) -> np.ndarray:
    any_threshold, substantial_threshold, mostly_threshold = thresholds
    pred = np.zeros(len(any_prob), dtype=np.int64)
    covered = any_prob >= any_threshold
    substantial = covered & (substantial_prob >= substantial_threshold)
    mostly = substantial & (mostly_prob >= mostly_threshold)
    pred[covered] = 1
    pred[substantial] = 2
    pred[mostly] = 3
    return pred


def covered_internal_label_ids(
    any_prob: np.ndarray,
    covered_internal_probs: np.ndarray,
    any_threshold: float,
) -> np.ndarray:
    any_threshold = min(1.0, max(0.0, float(any_threshold)))
    pred = np.zeros(len(any_prob), dtype=np.int64)
    covered = any_prob >= any_threshold
    if covered.any():
        pred[covered] = 1 + np.argmax(covered_internal_probs[covered], axis=1).astype(np.int64)
    return pred


def tune_covered_internal_threshold(
    y_true: np.ndarray,
    any_prob: np.ndarray,
    covered_internal_probs: np.ndarray,
    objective: str,
    steps: int,
    initial_threshold: float,
    any_range: tuple[float, float],
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> tuple[float, dict[str, Any]]:
    best_threshold = min(1.0, max(0.0, float(initial_threshold)))
    best_metrics = enriched_label_metrics(
        y_true,
        covered_internal_label_ids(any_prob, covered_internal_probs, best_threshold),
    )
    best_score = constrained_threshold_objective(
        best_metrics,
        objective,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    candidates = candidate_thresholds(any_prob, steps, *any_range)
    candidates = include_candidate(candidates, best_threshold, *any_range)
    for any_threshold in candidates:
        pred = covered_internal_label_ids(any_prob, covered_internal_probs, float(any_threshold))
        metrics = enriched_label_metrics(y_true, pred)
        current = constrained_threshold_objective(
            metrics,
            objective,
            min_not_recall,
            min_weak_recall,
            min_partial_recall,
            min_mostly_recall,
            min_covered_recall,
            max_partial_pred_ratio,
            max_mostly_pred_ratio,
            max_partial_pred_multiplier,
            max_mostly_pred_multiplier,
        )
        if current > best_score:
            best_score = current
            best_threshold = float(any_threshold)
            best_metrics = metrics
    best_metrics["threshold_objective"] = objective
    best_metrics["threshold_objective_score"] = round(float(best_score), 6)
    best_metrics["threshold_constraints"] = constraint_summary(
        best_metrics,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    return best_threshold, best_metrics


def tune_conditional_thresholds(
    y_true: np.ndarray,
    any_prob: np.ndarray,
    substantial_prob: np.ndarray,
    mostly_prob: np.ndarray,
    objective: str,
    steps: int,
    initial_thresholds: tuple[float, float, float],
    any_range: tuple[float, float],
    substantial_range: tuple[float, float],
    mostly_range: tuple[float, float],
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    best_thresholds = initial_thresholds
    best_metrics = enriched_label_metrics(
        y_true,
        conditional_label_ids(any_prob, substantial_prob, mostly_prob, best_thresholds),
    )
    best_score = constrained_threshold_objective(
        best_metrics,
        objective,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    any_candidates = candidate_thresholds(any_prob, steps, *any_range)
    substantial_candidates = candidate_thresholds(substantial_prob, steps, *substantial_range)
    mostly_candidates = candidate_thresholds(mostly_prob, steps, *mostly_range)
    any_candidates = include_candidate(any_candidates, initial_thresholds[0], *any_range)
    substantial_candidates = include_candidate(substantial_candidates, initial_thresholds[1], *substantial_range)
    mostly_candidates = include_candidate(mostly_candidates, initial_thresholds[2], *mostly_range)
    for any_threshold in any_candidates:
        for substantial_threshold in substantial_candidates:
            for mostly_threshold in mostly_candidates:
                thresholds = (
                    float(any_threshold),
                    float(substantial_threshold),
                    float(mostly_threshold),
                )
                pred = conditional_label_ids(any_prob, substantial_prob, mostly_prob, thresholds)
                metrics = enriched_label_metrics(y_true, pred)
                current = constrained_threshold_objective(
                    metrics,
                    objective,
                    min_not_recall,
                    min_weak_recall,
                    min_partial_recall,
                    min_mostly_recall,
                    min_covered_recall,
                    max_partial_pred_ratio,
                    max_mostly_pred_ratio,
                    max_partial_pred_multiplier,
                    max_mostly_pred_multiplier,
                )
                if current > best_score:
                    best_score = current
                    best_thresholds = thresholds
                    best_metrics = metrics
    best_metrics["threshold_objective"] = objective
    best_metrics["threshold_objective_score"] = round(float(best_score), 6)
    best_metrics["threshold_constraints"] = constraint_summary(
        best_metrics,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    return best_thresholds, best_metrics


def blend_label_ids(hierarchical_probs: np.ndarray, class_probs: np.ndarray, class_weight: float) -> np.ndarray:
    weight = min(1.0, max(0.0, float(class_weight)))
    probs = (1.0 - weight) * hierarchical_probs + weight * class_probs
    denom = np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    return np.argmax(probs / denom, axis=1).astype(np.int64)


def tune_hier_class_blend(
    y_true: np.ndarray,
    hierarchical_probs: np.ndarray,
    class_probs: np.ndarray,
    objective: str,
    steps: int,
    initial_weight: float,
    min_not_recall: float,
    min_weak_recall: float,
    min_partial_recall: float,
    min_mostly_recall: float,
    min_covered_recall: float,
    max_partial_pred_ratio: float,
    max_mostly_pred_ratio: float,
    max_partial_pred_multiplier: float,
    max_mostly_pred_multiplier: float,
) -> tuple[float, dict[str, Any]]:
    steps = max(2, int(steps))
    candidates = sorted(set(round(float(x), 6) for x in np.linspace(0.0, 1.0, steps)))
    candidates = include_candidate(candidates, float(initial_weight), 0.0, 1.0)
    best_weight = min(1.0, max(0.0, float(initial_weight)))
    best_metrics = enriched_label_metrics(y_true, blend_label_ids(hierarchical_probs, class_probs, best_weight))
    best_score = constrained_threshold_objective(
        best_metrics,
        objective,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    for class_weight in candidates:
        pred = blend_label_ids(hierarchical_probs, class_probs, class_weight)
        metrics = enriched_label_metrics(y_true, pred)
        current = constrained_threshold_objective(
            metrics,
            objective,
            min_not_recall,
            min_weak_recall,
            min_partial_recall,
            min_mostly_recall,
            min_covered_recall,
            max_partial_pred_ratio,
            max_mostly_pred_ratio,
            max_partial_pred_multiplier,
            max_mostly_pred_multiplier,
        )
        if current > best_score:
            best_score = current
            best_weight = float(class_weight)
            best_metrics = metrics
    best_metrics["blend_objective"] = objective
    best_metrics["blend_objective_score"] = round(float(best_score), 6)
    best_metrics["threshold_constraints"] = constraint_summary(
        best_metrics,
        min_not_recall,
        min_weak_recall,
        min_partial_recall,
        min_mostly_recall,
        min_covered_recall,
        max_partial_pred_ratio,
        max_mostly_pred_ratio,
        max_partial_pred_multiplier,
        max_mostly_pred_multiplier,
    )
    return best_weight, best_metrics


def prediction_probability_matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [
            [safe_float(dict(row.get(key) or {}).get(label), 0.0) for label in JOINT_LABELS]
            for row in rows
        ],
        dtype=np.float32,
    )


def covered_internal_probability_matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [
            [safe_float(dict(row.get(key) or {}).get(label), 0.0) for label in COVERED_INTERNAL_LABELS]
            for row in rows
        ],
        dtype=np.float32,
    )


def first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def graph_paper_date(paper_id: str) -> int | None:
    match = re.match(r"^(\d{2})(\d{2})", str(paper_id or ""))
    if not match:
        return None
    yy, mm = int(match.group(1)), int(match.group(2))
    return (2000 + yy if yy < 90 else 1900 + yy) * 100 + mm


def graph_idea_date(row: dict[str, Any]) -> int | None:
    dates = [graph_paper_date(str(paper)) for paper in row.get("papers", [])]
    dates = [date for date in dates if date is not None]
    return min(dates) if dates else None


class GraphStructureContext:

    def __init__(
        self,
        canonical_ideas: Path | None,
        graph_file: Path | None,
        relations: set[str] | None = None,
    ) -> None:
        self.relations = relations or set(GRAPH_RELATIONS)
        self.idea_dates: dict[str, int] = {}
        self.idea_text: dict[str, str] = {}
        self.idea_meta: dict[str, dict[str, Any]] = {}
        self.adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.direct: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.loaded = False
        self.edge_count = 0
        self.relation_counts: Counter[str] = Counter()
        self.feature_names = self._build_feature_names()

        if canonical_ideas and canonical_ideas.exists():
            ideas = json.loads(canonical_ideas.read_text(encoding="utf-8"))
            for row in ideas:
                idea_id = str(row.get("canonical_idea_id") or "")
                if not idea_id:
                    continue
                date = graph_idea_date(row)
                if date:
                    self.idea_dates[idea_id] = int(date)
                self.idea_text[idea_id] = str(row.get("text") or "")
                self.idea_meta[idea_id] = {
                    "primary_aspect": str(row.get("primary_aspect") or ""),
                    "contribution_type": str(row.get("contribution_type") or ""),
                    "graph_form": str(row.get("graph_form") or ""),
                    "task_family": str(row.get("task_family") or ""),
                    "mechanism_family": str(row.get("mechanism_family") or ""),
                    "size": safe_float(row.get("size"), 0.0),
                    "paper_count": len(row.get("papers") or []),
                    "raw_idea_count": len(row.get("raw_ids") or []),
                }

        if graph_file and graph_file.exists():
            graph = json.loads(graph_file.read_text(encoding="utf-8"))
            for edge in graph.get("links", graph.get("edges", [])):
                relation = str(edge.get("relation") or "")
                source = str(edge.get("source") or "")
                target = str(edge.get("target") or "")
                if relation not in self.relations:
                    continue
                if not source.startswith("CI_") or not target.startswith("CI_") or source == target:
                    continue
                weight = safe_float(edge.get("weight"), 1.0)
                self._add_edge(source, target, relation, weight, "out")
                self._add_edge(target, source, relation, weight, "in")
                self.direct[(source, target)].append(f"out_{relation}")
                self.direct[(target, source)].append(f"in_{relation}")
                self.edge_count += 1
                self.relation_counts[relation] += 1
        self.loaded = bool(self.idea_text or self.edge_count)

    def _add_edge(self, source: str, target: str, relation: str, weight: float, direction: str) -> None:
        self.adjacency[source].append(
            {
                "target": target,
                "relation": relation,
                "direction": direction,
                "token": f"{direction}_{relation}",
                "weight": float(weight),
            }
        )

    def is_historical(self, idea_id: str, target_date: int) -> bool:
        date = self.idea_dates.get(idea_id)
        return not target_date or not date or date < target_date

    def neighbor_edges(self, idea_id: str, target_date: int, excluded: set[str]) -> list[dict[str, Any]]:
        edges = []
        for edge in self.adjacency.get(idea_id, []):
            neighbor = str(edge["target"])
            if neighbor in excluded or not self.is_historical(neighbor, target_date):
                continue
            edges.append(edge)
        edges.sort(
            key=lambda edge: (
                RELATION_PRIORITY.get(str(edge["relation"]), 0),
                safe_float(edge.get("weight"), 0.0),
                str(edge.get("target") or ""),
            ),
            reverse=True,
        )
        return edges

    def direct_relations(self, source: str, target: str) -> list[str]:
        return sorted(set(self.direct.get((source, target), [])))

    @staticmethod
    def token_relation(token: str) -> str:
        return str(token).split("_", 1)[1] if "_" in str(token) else str(token)

    @staticmethod
    def _build_feature_names() -> list[str]:
        names = []
        names.extend(f"target_direct_{token}" for token in GRAPH_RELATION_TOKENS)
        names.extend(f"top_prior_direct_{token}" for token in GRAPH_RELATION_TOKENS)
        names.extend(f"one_hop_{token}" for token in GRAPH_RELATION_TOKENS)
        names.extend(f"two_hop_{token}" for token in GRAPH_PATH_TOKENS)
        names.extend(f"two_hop_top_prior_hit_{token}" for token in GRAPH_RELATION_TOKENS)
        names.extend(
            [
                "one_hop_degree",
                "two_hop_path_total",
                "top_prior_direct_density",
                "two_hop_top_prior_hit_total",
            ]
        )
        names.extend(f"target_same_{field}" for field in GRAPH_FACET_FIELDS)
        names.extend(f"top_prior_same_{field}_ratio" for field in GRAPH_FACET_FIELDS)
        names.extend(f"one_hop_same_{field}_ratio" for field in GRAPH_FACET_FIELDS)
        names.extend(f"one_hop_target_same_{field}_ratio" for field in GRAPH_FACET_FIELDS)
        names.extend(
            [
                "prior_size",
                "prior_paper_count",
                "prior_raw_idea_count",
                "prior_age_months",
                "prior_recent_24m",
                "prior_recent_60m",
                "top_prior_connected_count",
                "top_prior_connected_ratio",
                "top_prior_constructive_relation_ratio",
                "top_prior_contrast_relation_ratio",
                "top_prior_relation_entropy",
                "top_prior_neighbor_overlap_max",
                "top_prior_neighbor_overlap_mean",
                "top_prior_shared_neighbor_total",
                "top_prior_shared_neighbor_ratio",
                "one_hop_out_ratio",
                "one_hop_in_ratio",
                "one_hop_constructive_relation_ratio",
                "one_hop_contrast_relation_ratio",
                "one_hop_relation_entropy",
                "one_hop_mean_weight",
                "one_hop_max_weight",
                "one_hop_recent_24m_ratio",
                "one_hop_recent_60m_ratio",
                "one_hop_age_mean_months",
                "one_hop_age_min_months",
                "two_hop_unique_mid_count",
                "two_hop_unique_second_count",
                "two_hop_constructive_path_ratio",
                "two_hop_contrast_path_ratio",
                "two_hop_relation_entropy",
                "two_hop_second_top_prior_ratio",
            ]
        )
        return names

    def zero_features(self) -> list[float]:
        return [0.0 for _ in self.feature_names]

    def features_for_prior(
        self,
        target_id: str,
        prior_id: str,
        top_prior_ids: list[str],
        target_date: int,
        max_one_hop: int,
        max_two_hop_per_neighbor: int,
        include_direct_target_prior: bool,
    ) -> tuple[list[float], dict[str, Any]]:
        values = {name: 0.0 for name in self.feature_names}
        if not prior_id:
            return self.zero_features(), {
                "graph_features_found": False,
                "graph_feature_nonzero_count": 0,
                "graph_feature_l1": 0.0,
            }

        top_prior_set = {pid for pid in top_prior_ids if pid and pid != prior_id}
        excluded = {prior_id, target_id}
        prior_meta = self.idea_meta.get(prior_id, {})
        target_meta = self.idea_meta.get(target_id, {})
        prior_date = self.idea_dates.get(prior_id)
        prior_age = month_gap(prior_date, target_date)

        values["prior_size"] = scaled_count(safe_float(prior_meta.get("size"), 0.0), 8.0)
        values["prior_paper_count"] = scaled_count(safe_float(prior_meta.get("paper_count"), 0.0), 4.0)
        values["prior_raw_idea_count"] = scaled_count(safe_float(prior_meta.get("raw_idea_count"), 0.0), 8.0)
        if prior_age is not None:
            values["prior_age_months"] = scaled_count(prior_age, 120.0)
            values["prior_recent_24m"] = max(0.0, 1.0 - min(24.0, float(prior_age)) / 24.0)
            values["prior_recent_60m"] = max(0.0, 1.0 - min(60.0, float(prior_age)) / 60.0)

        for field in GRAPH_FACET_FIELDS:
            prior_value = str(prior_meta.get(field) or "")
            target_value = str(target_meta.get(field) or "")
            if prior_value and target_value:
                values[f"target_same_{field}"] = 1.0 if prior_value == target_value else 0.0

        if include_direct_target_prior and target_id:
            for token in self.direct_relations(target_id, prior_id):
                key = f"target_direct_{token}"
                if key in values:
                    values[key] = 1.0

        top_direct_counts: Counter[str] = Counter()
        top_connected_count = 0
        top_same_counts: Counter[str] = Counter()
        for other_id in top_prior_ids:
            if not other_id or other_id == prior_id:
                continue
            rels = self.direct_relations(prior_id, other_id)
            if rels:
                top_connected_count += 1
                top_direct_counts.update(rels)
            other_meta = self.idea_meta.get(other_id, {})
            for field in GRAPH_FACET_FIELDS:
                prior_value = str(prior_meta.get(field) or "")
                other_value = str(other_meta.get(field) or "")
                if prior_value and other_value and prior_value == other_value:
                    top_same_counts[field] += 1
        top_den = max(1, len(top_prior_set))
        for token, count in top_direct_counts.items():
            key = f"top_prior_direct_{token}"
            if key in values:
                values[key] = min(1.0, float(count) / float(top_den))
        top_relation_total = sum(top_direct_counts.values())
        top_constructive_total = sum(
            count for token, count in top_direct_counts.items()
            if self.token_relation(token) in GRAPH_CONSTRUCTIVE_RELATIONS
        )
        top_contrast_total = sum(
            count for token, count in top_direct_counts.items()
            if self.token_relation(token) in GRAPH_CONTRAST_RELATIONS
        )
        values["top_prior_direct_density"] = safe_ratio(top_relation_total, top_den)
        values["top_prior_connected_count"] = scaled_count(top_connected_count, 4.0)
        values["top_prior_connected_ratio"] = safe_ratio(top_connected_count, top_den)
        values["top_prior_constructive_relation_ratio"] = safe_ratio(top_constructive_total, top_relation_total)
        values["top_prior_contrast_relation_ratio"] = safe_ratio(top_contrast_total, top_relation_total)
        values["top_prior_relation_entropy"] = normalized_entropy(top_direct_counts, len(GRAPH_RELATION_TOKENS))
        for field in GRAPH_FACET_FIELDS:
            values[f"top_prior_same_{field}_ratio"] = safe_ratio(top_same_counts[field], top_den)

        one_hop_all = self.neighbor_edges(prior_id, target_date, excluded)
        one_hop = one_hop_all[: max(0, max_one_hop)]
        one_counts = Counter(str(edge["token"]) for edge in one_hop_all)
        for token, count in one_counts.items():
            key = f"one_hop_{token}"
            if key in values:
                values[key] = scaled_count(count, 20.0)
        values["one_hop_degree"] = scaled_count(len(one_hop_all), 40.0)
        one_degree = len(one_hop_all)
        one_relation_counts = Counter(str(edge["relation"]) for edge in one_hop_all)
        one_out = sum(1 for edge in one_hop_all if str(edge.get("direction")) == "out")
        one_in = sum(1 for edge in one_hop_all if str(edge.get("direction")) == "in")
        one_constructive = sum(
            1 for edge in one_hop_all
            if str(edge.get("relation")) in GRAPH_CONSTRUCTIVE_RELATIONS
        )
        one_contrast = sum(
            1 for edge in one_hop_all
            if str(edge.get("relation")) in GRAPH_CONTRAST_RELATIONS
        )
        one_weights = [safe_float(edge.get("weight"), 0.0) for edge in one_hop_all]
        values["one_hop_out_ratio"] = safe_ratio(one_out, one_degree)
        values["one_hop_in_ratio"] = safe_ratio(one_in, one_degree)
        values["one_hop_constructive_relation_ratio"] = safe_ratio(one_constructive, one_degree)
        values["one_hop_contrast_relation_ratio"] = safe_ratio(one_contrast, one_degree)
        values["one_hop_relation_entropy"] = normalized_entropy(one_relation_counts, len(GRAPH_RELATIONS))
        if one_weights:
            values["one_hop_mean_weight"] = min(1.0, max(0.0, float(np.mean(one_weights))))
            values["one_hop_max_weight"] = min(1.0, max(0.0, float(max(one_weights))))

        one_neighbor_ids = {str(edge["target"]) for edge in one_hop_all}
        one_neighbor_ages = [
            gap for gap in (month_gap(self.idea_dates.get(neighbor), target_date) for neighbor in one_neighbor_ids)
            if gap is not None
        ]
        if one_neighbor_ages:
            values["one_hop_recent_24m_ratio"] = safe_ratio(sum(1 for gap in one_neighbor_ages if gap <= 24), one_degree)
            values["one_hop_recent_60m_ratio"] = safe_ratio(sum(1 for gap in one_neighbor_ages if gap <= 60), one_degree)
            values["one_hop_age_mean_months"] = scaled_count(float(np.mean(one_neighbor_ages)), 120.0)
            values["one_hop_age_min_months"] = scaled_count(min(one_neighbor_ages), 120.0)

        one_same_prior: Counter[str] = Counter()
        one_same_target: Counter[str] = Counter()
        for neighbor in one_neighbor_ids:
            neighbor_meta = self.idea_meta.get(neighbor, {})
            for field in GRAPH_FACET_FIELDS:
                prior_value = str(prior_meta.get(field) or "")
                target_value = str(target_meta.get(field) or "")
                neighbor_value = str(neighbor_meta.get(field) or "")
                if neighbor_value and prior_value and neighbor_value == prior_value:
                    one_same_prior[field] += 1
                if neighbor_value and target_value and neighbor_value == target_value:
                    one_same_target[field] += 1
        for field in GRAPH_FACET_FIELDS:
            values[f"one_hop_same_{field}_ratio"] = safe_ratio(one_same_prior[field], one_degree)
            values[f"one_hop_target_same_{field}_ratio"] = safe_ratio(one_same_target[field], one_degree)

        overlap_ratios: list[float] = []
        shared_neighbor_total = 0
        for other_id in top_prior_set:
            other_neighbors = {
                str(edge["target"])
                for edge in self.neighbor_edges(other_id, target_date, {other_id, target_id})
            }
            if not one_neighbor_ids and not other_neighbors:
                continue
            shared = len(one_neighbor_ids & other_neighbors)
            union = len(one_neighbor_ids | other_neighbors)
            shared_neighbor_total += shared
            overlap_ratios.append(safe_ratio(shared, union))
        if overlap_ratios:
            values["top_prior_neighbor_overlap_max"] = max(overlap_ratios)
            values["top_prior_neighbor_overlap_mean"] = float(np.mean(overlap_ratios))
        values["top_prior_shared_neighbor_total"] = scaled_count(shared_neighbor_total, 20.0)
        values["top_prior_shared_neighbor_ratio"] = safe_ratio(
            shared_neighbor_total,
            max(1, len(one_neighbor_ids) * len(top_prior_set)),
        )

        path_counts: Counter[str] = Counter()
        hit_counts: Counter[str] = Counter()
        mid_ids: set[str] = set()
        second_ids: set[str] = set()
        constructive_paths = 0
        contrast_paths = 0
        for edge in one_hop:
            mid = str(edge["target"])
            mid_ids.add(mid)
            second_edges = self.neighbor_edges(mid, target_date, excluded | {mid})[: max(0, max_two_hop_per_neighbor)]
            for edge2 in second_edges:
                second = str(edge2["target"])
                if second == prior_id or second == target_id:
                    continue
                path_key = f"{edge['token']}>{edge2['token']}"
                path_counts[path_key] += 1
                second_ids.add(second)
                edge_rel = str(edge.get("relation") or "")
                edge2_rel = str(edge2.get("relation") or "")
                if edge_rel in GRAPH_CONSTRUCTIVE_RELATIONS or edge2_rel in GRAPH_CONSTRUCTIVE_RELATIONS:
                    constructive_paths += 1
                if edge_rel in GRAPH_CONTRAST_RELATIONS or edge2_rel in GRAPH_CONTRAST_RELATIONS:
                    contrast_paths += 1
                if second in top_prior_set:
                    hit_counts[str(edge2["token"])] += 1
        for token, count in path_counts.items():
            key = f"two_hop_{token}"
            if key in values:
                values[key] = scaled_count(count, 10.0)
        for token, count in hit_counts.items():
            key = f"two_hop_top_prior_hit_{token}"
            if key in values:
                values[key] = scaled_count(count, 5.0)
        path_total = sum(path_counts.values())
        top_hit_total = sum(hit_counts.values())
        values["two_hop_path_total"] = scaled_count(path_total, 50.0)
        values["two_hop_top_prior_hit_total"] = scaled_count(top_hit_total, 10.0)
        values["two_hop_unique_mid_count"] = scaled_count(len(mid_ids), 32.0)
        values["two_hop_unique_second_count"] = scaled_count(len(second_ids), 80.0)
        values["two_hop_constructive_path_ratio"] = safe_ratio(constructive_paths, path_total)
        values["two_hop_contrast_path_ratio"] = safe_ratio(contrast_paths, path_total)
        values["two_hop_relation_entropy"] = normalized_entropy(path_counts, len(GRAPH_PATH_TOKENS))
        values["two_hop_second_top_prior_ratio"] = safe_ratio(len(second_ids & top_prior_set), max(1, len(top_prior_set)))

        vector = [float(values[name]) for name in self.feature_names]
        nonzero = sum(1 for value in vector if abs(value) > 1e-8)
        return vector, {
            "graph_features_found": nonzero > 0,
            "graph_feature_nonzero_count": int(nonzero),
            "graph_feature_l1": round(float(sum(abs(value) for value in vector)), 6),
        }

def prior_text(target_id: str, prior: dict[str, Any], text_lookup: TextLookup) -> str:
    direct = str(prior.get("prior_text") or "")
    if direct:
        return direct
    prior_id = str(prior.get("prior_idea_id") or "")
    return text_lookup.target_prior_text.get((target_id, prior_id)) or text_lookup.idea_text.get(prior_id, "")


def prior_meta(target_id: str, prior: dict[str, Any], text_lookup: TextLookup) -> dict[str, Any]:
    prior_id = str(prior.get("prior_idea_id") or "")
    return text_lookup.target_prior_meta.get((target_id, prior_id), {})


def signal_from_prior(
    prior: dict[str, Any],
    meta: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    value = first_present(prior, keys)
    if value is not None:
        return value
    return first_present(meta, keys, default)


def effective_joint_contribution(label: Any, score: Any) -> float:
    normalized_label = str(label or "")
    normalized_score = min(1.0, max(0.0, safe_float(score, 0.0)))
    if normalized_label == "large_cover":
        return max(0.72, normalized_score)
    if normalized_label == "partial_cover":
        return min(max(normalized_score, 0.24), 0.38)
    if normalized_label == "related_not_covering":
        return min(normalized_score, 0.04)
    return 0.0


def pair_prompt(
    row: dict[str, Any],
    novelty: dict[str, Any],
    prior: dict[str, Any],
    text_lookup: TextLookup,
    rank: int,
    include_pred_pair_signals: bool,
    include_gold_pair_signals: bool,
) -> tuple[str, dict[str, Any]]:
    target_id = str(row.get("target_idea_id") or "")
    prior_id = str(prior.get("prior_idea_id") or "")
    meta = prior_meta(target_id, prior, text_lookup)
    target_text = str(row.get("target_text") or novelty.get("target_text") or text_lookup.idea_text.get(target_id, ""))
    prior_body = prior_text(target_id, prior, text_lookup)

    signal_parts = []
    pred_label = signal_from_prior(prior, meta, PRED_PAIR_LABEL_KEYS, "unknown")
    pred_score = safe_float(signal_from_prior(prior, meta, PRED_PAIR_SCORE_KEYS, 0.0), 0.0)
    if include_pred_pair_signals:
        signal_parts.extend(
            [
                f"pred_pair_label={pred_label}",
                f"pred_pair_score={pred_score:.3f}",
                f"rank_score={safe_float(prior.get('rank_score'), pred_score):.3f}",
            ]
        )
    if include_gold_pair_signals:
        gold_label = signal_from_prior(prior, meta, GOLD_PAIR_LABEL_KEYS, "unknown")
        gold_score = safe_float(signal_from_prior(prior, meta, GOLD_PAIR_SCORE_KEYS, 0.0), 0.0)
        signal_parts.extend([f"gold_pair_label={gold_label}", f"gold_pair_score={gold_score:.3f}"])

    prior_meta_parts = []
    for key, label in [
        ("prior_date", "date"),
        ("prior_primary_aspect", "aspect"),
        ("prior_contribution_type", "contribution"),
        ("prior_task_family", "task"),
    ]:
        if meta.get(key) is not None:
            prior_meta_parts.append(f"{label}={meta[key]}")

    signal_text = "; ".join(signal_parts) if signal_parts else "pair_signals=none"
    prior_meta_text = "; ".join(prior_meta_parts) if prior_meta_parts else "prior_metadata=unknown"
    prompt = (
        "Judge how much this single historical prior covers the target scientific idea.\n"
        "Use semantic content and publication context. Relatedness alone is not coverage.\n"
        f"[TARGET METADATA] {metadata_line(row, novelty)}\n"
        f"[TARGET IDEA]\n{target_text}\n"
        f"[PRIOR {rank} METADATA] id={prior_id}; {signal_text}; {prior_meta_text}\n"
        f"[HISTORICAL PRIOR]\n{prior_body}"
    )
    prompt_meta = {
        "rank": rank,
        "prior_idea_id": prior_id,
        "prior_text_found": bool(prior_body),
        "pred_pair_label": pred_label,
        "pred_pair_score": round(float(pred_score), 6),
        "rank_score": round(float(safe_float(prior.get("rank_score"), pred_score)), 6),
    }
    return prompt, prompt_meta


def pair_aux_target(
    prior: dict[str, Any],
    meta: dict[str, Any],
    mode: str,
) -> tuple[float, float]:
    if mode == "none":
        return 0.0, 0.0
    if mode == "gold_effective":
        label = signal_from_prior(prior, meta, GOLD_PAIR_LABEL_KEYS)
        value = signal_from_prior(prior, meta, GOLD_PAIR_SCORE_KEYS)
        if label is None or label == "" or value is None or value == "":
            return 0.0, 0.0
        return effective_joint_contribution(label, value), 1.0
    if mode == "gold":
        value = signal_from_prior(prior, meta, GOLD_PAIR_SCORE_KEYS)
    elif mode == "pred":
        value = signal_from_prior(prior, meta, PRED_PAIR_SCORE_KEYS)
    else:
        raise ValueError(f"unknown pair aux target mode: {mode}")
    if value is None or value == "":
        return 0.0, 0.0
    return min(1.0, max(0.0, safe_float(value, 0.0))), 1.0


class JointMilDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        novelty_index: dict[str, dict[str, Any]],
        text_lookup: TextLookup,
        top_k: int,
        include_pred_pair_signals: bool,
        include_gold_pair_signals: bool,
        pair_aux_target_mode: str,
        graph_context: GraphStructureContext | None = None,
        graph_feature_max_one_hop: int = 32,
        graph_feature_max_two_hop_per_neighbor: int = 16,
        include_direct_target_prior_graph_relations: bool = False,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.top_k = top_k
        self.graph_feature_dim = len(graph_context.feature_names) if graph_context is not None else 0
        missing_text = 0
        missing_pair_targets = 0
        missing_graph_context = 0
        for row in rows:
            target_id = str(row.get("target_idea_id") or "")
            novelty = novelty_index.get(target_id)
            if not novelty:
                continue
            label = str(novelty.get("joint_coverage_label") or "")
            if label not in JOINT_TO_ID:
                continue
            priors = ranking_priors(row)[:top_k]
            top_prior_ids = [str(prior.get("prior_idea_id") or "") for prior in priors]
            target_date = int(row.get("target_date") or novelty.get("target_date") or 0)
            pairs = []
            prompt_priors = []
            pair_targets = []
            pair_target_mask = []
            pair_mask = []
            graph_features = []
            for idx in range(top_k):
                if idx < len(priors):
                    prior = priors[idx]
                    meta = prior_meta(target_id, prior, text_lookup)
                    graph_feature_vector: list[float] = []
                    graph_feature_meta = {
                        "graph_features_found": False,
                        "graph_feature_nonzero_count": 0,
                        "graph_feature_l1": 0.0,
                    }
                    if graph_context is not None:
                        graph_feature_vector, graph_feature_meta = graph_context.features_for_prior(
                            target_id=target_id,
                            prior_id=str(prior.get("prior_idea_id") or ""),
                            top_prior_ids=top_prior_ids,
                            target_date=target_date,
                            max_one_hop=graph_feature_max_one_hop,
                            max_two_hop_per_neighbor=graph_feature_max_two_hop_per_neighbor,
                            include_direct_target_prior=include_direct_target_prior_graph_relations,
                        )
                    prompt, prompt_meta = pair_prompt(
                        row,
                        novelty,
                        prior,
                        text_lookup,
                        idx + 1,
                        include_pred_pair_signals,
                        include_gold_pair_signals,
                    )
                    prompt_meta.update(graph_feature_meta)
                    target, target_mask = pair_aux_target(prior, meta, pair_aux_target_mode)
                    missing_text += 0 if prompt_meta["prior_text_found"] else 1
                    missing_pair_targets += 0 if target_mask else 1
                    missing_graph_context += 0 if prompt_meta["graph_features_found"] else 1
                    pairs.append(prompt)
                    prompt_priors.append(prompt_meta)
                    pair_targets.append(target)
                    pair_target_mask.append(target_mask)
                    pair_mask.append(1.0)
                    graph_features.append(graph_feature_vector)
                else:
                    pairs.append(
                        "No historical prior candidate is available for this rank.\n"
                        f"[TARGET METADATA] {metadata_line(row, novelty)}\n"
                        f"[TARGET IDEA]\n{novelty.get('target_text') or ''}"
                    )
                    prompt_priors.append(
                        {
                            "rank": idx + 1,
                            "prior_idea_id": "",
                            "prior_text_found": False,
                            "pred_pair_label": "missing",
                            "pred_pair_score": 0.0,
                            "rank_score": 0.0,
                            "graph_features_found": False,
                            "graph_feature_nonzero_count": 0,
                            "graph_feature_l1": 0.0,
                        }
                    )
                    pair_targets.append(0.0)
                    pair_target_mask.append(0.0)
                    pair_mask.append(0.0)
                    graph_features.append(graph_context.zero_features() if graph_context is not None else [])

            self.items.append(
                {
                    "target_idea_id": target_id,
                    "pair_prompts": pairs,
                    "prompt_priors": prompt_priors,
                    "ranking": row,
                    "label": JOINT_TO_ID[label],
                    "score": safe_float(novelty.get("joint_coverage_score"), 0.0),
                    "weight": min(1.0, max(0.2, safe_float(novelty.get("label_confidence"), 1.0))),
                    "pair_score_target": pair_targets,
                    "pair_target_mask": pair_target_mask,
                    "pair_mask": pair_mask,
                    "graph_features": graph_features,
                }
            )
        if not self.items:
            raise ValueError("No joinable MIL joint coverage rows")
        self.missing_prior_texts = missing_text
        self.missing_pair_targets = missing_pair_targets
        self.missing_graph_context = missing_graph_context

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class JointMilCollator:
    def __init__(self, tokenizer: Any, max_length: int, top_k: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k = top_k

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        flat_prompts = [prompt for item in batch for prompt in item["pair_prompts"]]
        encoded = self.tokenizer(
            flat_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.reshape(batch_size, self.top_k, -1)
            for key, value in encoded.items()
        }
        return {
            "inputs": inputs,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "score": torch.tensor([item["score"] for item in batch], dtype=torch.float32),
            "weight": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
            "pair_score_target": torch.tensor([item["pair_score_target"] for item in batch], dtype=torch.float32),
            "pair_target_mask": torch.tensor([item["pair_target_mask"] for item in batch], dtype=torch.float32),
            "pair_mask": torch.tensor([item["pair_mask"] for item in batch], dtype=torch.float32),
            "graph_features": torch.tensor([item["graph_features"] for item in batch], dtype=torch.float32),
            "target_idea_id": [item["target_idea_id"] for item in batch],
            "ranking": [item["ranking"] for item in batch],
            "prompt_priors": [item["prompt_priors"] for item in batch],
        }


@dataclass
class JointMilConfig:
    model_name: str
    dropout: float = 0.15
    noisy_or_blend_weight: float = 0.25
    score_prediction_mode: str = "hierarchical"
    score_blend_weight: float = 0.30
    graph_feature_dim: int = 0
    graph_feature_gate_init: float = -3.0
    class_score_means: list[float] = field(default_factory=lambda: list(DEFAULT_CLASS_SCORE_MEANS))
    trust_remote_code: bool = True
    cache_dir: str | None = None
    local_files_only: bool = False


class JointMilModel(nn.Module):
    def __init__(self, cfg: JointMilConfig) -> None:
        super().__init__()
        self.cfg = cfg
        kwargs: dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "cache_dir": cfg.cache_dir,
            "local_files_only": cfg.local_files_only,
        }
        self.config = AutoConfig.from_pretrained(cfg.model_name, **kwargs)
        self.backbone = AutoModel.from_pretrained(cfg.model_name, **kwargs)
        hidden_size = int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))
        if not hidden_size:
            raise ValueError(f"Cannot infer hidden size for {cfg.model_name}")
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "class_score_means",
            torch.tensor(clamp_monotonic(cfg.class_score_means), dtype=torch.float32),
        )
        self.pair_score_head = nn.Linear(hidden_size, 1)
        self.attention_head = nn.Linear(hidden_size, 1)
        self.graph_feature_dim = max(0, int(cfg.graph_feature_dim))
        if self.graph_feature_dim:
            self.graph_feature_norm = nn.LayerNorm(self.graph_feature_dim)
            self.graph_feature_proj = nn.Sequential(
                nn.Linear(self.graph_feature_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.graph_feature_gate_head = nn.Linear(self.graph_feature_dim, 1)
            nn.init.zeros_(self.graph_feature_gate_head.weight)
            nn.init.zeros_(self.graph_feature_gate_head.bias)
            self.graph_feature_gate = nn.Parameter(torch.tensor(float(cfg.graph_feature_gate_init), dtype=torch.float32))
        else:
            self.graph_feature_norm = None
            self.graph_feature_proj = None
            self.graph_feature_gate_head = None
            self.graph_feature_gate = None
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_size * 2 + 4, hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.class_head = nn.Linear(hidden_size, len(JOINT_LABELS))
        self.ordinal_head = nn.Linear(hidden_size, len(JOINT_LABELS) - 1)
        self.any_head = nn.Linear(hidden_size, 1)
        self.substantial_head = nn.Linear(hidden_size, 1)
        self.mostly_head = nn.Linear(hidden_size, 1)
        self.covered_internal_head = nn.Linear(hidden_size, len(COVERED_INTERNAL_LABELS))
        self.score_head = nn.Linear(hidden_size, 1)

    def pooled(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        pair_mask: torch.Tensor,
        graph_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, top_k, seq_len = inputs["input_ids"].shape
        flat_inputs = {
            key: value.reshape(batch_size * top_k, seq_len)
            for key, value in inputs.items()
        }
        pair_hidden = self.dropout(self.pooled(flat_inputs)).reshape(batch_size, top_k, -1)
        valid = pair_mask.bool()
        graph_gate_value = pair_hidden.new_zeros(batch_size)
        if self.graph_feature_dim and graph_features is not None and graph_features.shape[-1] == self.graph_feature_dim:
            graph_values = graph_features.to(device=pair_hidden.device, dtype=pair_hidden.dtype)
            graph_norm = self.graph_feature_norm(graph_values)
            graph_hidden = self.graph_feature_proj(graph_norm)
            graph_gate_logits = self.graph_feature_gate.to(dtype=pair_hidden.dtype, device=pair_hidden.device)
            graph_gate_logits = graph_gate_logits + self.graph_feature_gate_head(graph_norm).squeeze(-1)
            pair_graph_gate = torch.sigmoid(graph_gate_logits)
            valid_for_graph = valid.to(pair_hidden.dtype)
            pair_hidden = pair_hidden + pair_graph_gate.unsqueeze(-1) * graph_hidden * valid_for_graph.unsqueeze(-1)
            graph_gate_value = (pair_graph_gate * valid_for_graph).sum(dim=1) / valid_for_graph.sum(dim=1).clamp_min(1.0)
        safe_valid = valid.clone()
        no_valid = ~safe_valid.any(dim=1)
        if no_valid.any():
            safe_valid[no_valid, 0] = True
        valid_float = valid.to(pair_hidden.dtype)
        denom = valid_float.sum(dim=1).clamp_min(1.0)

        pair_scores = torch.sigmoid(self.pair_score_head(pair_hidden).squeeze(-1).float())
        masked_pair_scores = pair_scores * valid_float
        noisy_or = 1.0 - torch.prod(1.0 - masked_pair_scores.clamp(max=0.99), dim=1)
        max_pair = masked_pair_scores.max(dim=1).values
        mean_pair = masked_pair_scores.sum(dim=1) / denom
        count_ratio = denom / float(top_k)

        attention_logits = self.attention_head(pair_hidden).squeeze(-1).float()
        attention_logits = attention_logits.masked_fill(~safe_valid, -1e4)
        attention = torch.softmax(attention_logits, dim=1) * safe_valid.to(attention_logits.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)

        attn_pool = torch.einsum("bk,bkh->bh", attention.to(pair_hidden.dtype), pair_hidden)
        masked_hidden = pair_hidden.masked_fill(~safe_valid.unsqueeze(-1), -1e4)
        max_pool = masked_hidden.max(dim=1).values
        summary = torch.stack([noisy_or, max_pair, mean_pair, count_ratio], dim=1).to(pair_hidden.dtype)
        bag = self.aggregator(torch.cat([attn_pool, max_pool, summary], dim=1))

        direct_score = torch.sigmoid(self.score_head(bag).squeeze(-1).float())
        ordinal_logits = self.ordinal_head(bag)
        any_logits = self.any_head(bag).squeeze(-1)
        substantial_logits = self.substantial_head(bag).squeeze(-1)
        mostly_logits = self.mostly_head(bag).squeeze(-1)
        covered_internal_logits = self.covered_internal_head(bag)
        hierarchy_probs = hierarchical_class_probs(any_logits, substantial_logits, mostly_logits)
        covered_internal_probs = covered_internal_class_probs(any_logits, covered_internal_logits)
        hierarchy_score = expected_score_from_probs(hierarchy_probs, self.class_score_means)
        covered_internal_score = expected_score_from_probs(covered_internal_probs, self.class_score_means)
        legacy_ordinal_probs = ordinal_class_probs(ordinal_logits)
        legacy_ordinal_score = expected_score_from_probs(legacy_ordinal_probs, self.class_score_means)

        mode = self.cfg.score_prediction_mode
        if mode == "direct":
            score = direct_score
        elif mode == "blend":
            weight = min(1.0, max(0.0, float(self.cfg.score_blend_weight)))
            score = (1.0 - weight) * hierarchy_score + weight * direct_score
        elif mode == "covered_internal":
            score = covered_internal_score
        elif mode == "noisy_or":
            score = noisy_or
        elif mode == "noisy_blend":
            weight = min(1.0, max(0.0, float(self.cfg.noisy_or_blend_weight)))
            score = (1.0 - weight) * direct_score + weight * noisy_or
        else:
            score = hierarchy_score
        return {
            "logits": self.class_head(bag),
            "ordinal_logits": ordinal_logits,
            "any_logits": any_logits,
            "substantial_logits": substantial_logits,
            "mostly_logits": mostly_logits,
            "covered_internal_logits": covered_internal_logits,
            "hierarchical_probs": hierarchy_probs,
            "covered_internal_probs": covered_internal_probs,
            "legacy_ordinal_probs": legacy_ordinal_probs,
            "score": score,
            "direct_score": direct_score,
            "bag_score": direct_score,
            "hierarchical_score": hierarchy_score,
            "covered_internal_score": covered_internal_score,
            "legacy_ordinal_score": legacy_ordinal_score,
            "mil_noisy_or_score": noisy_or,
            "pair_scores": pair_scores,
            "attention": attention,
            "graph_feature_gate": graph_gate_value,
        }


def move_inputs_to_device(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
    pair_mask = batch["pair_mask"].to(device)
    graph_features = batch["graph_features"].to(device)
    return inputs, pair_mask, graph_features


def probabilities(
    outputs: dict[str, torch.Tensor],
    prediction_mode: str,
    ordinal_blend_weight: float,
    hier_class_blend_weight: float,
    internal_blend_weight: float,
) -> torch.Tensor:
    if prediction_mode in {"hierarchical", "separate", "conditional"}:
        return outputs["hierarchical_probs"].float()
    if prediction_mode == "covered_internal":
        return outputs["covered_internal_probs"].float()
    class_probs = torch.softmax(outputs["logits"].float(), dim=-1)
    if prediction_mode == "class":
        return class_probs
    if prediction_mode == "hier_class_blend":
        weight = min(1.0, max(0.0, float(hier_class_blend_weight)))
        probs = (1.0 - weight) * outputs["hierarchical_probs"].float() + weight * class_probs
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if prediction_mode == "internal_blend":
        weight = min(1.0, max(0.0, float(internal_blend_weight)))
        probs = (1.0 - weight) * outputs["hierarchical_probs"].float() + weight * outputs["covered_internal_probs"].float()
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ordinal_probs = outputs.get("legacy_ordinal_probs")
    if ordinal_probs is None:
        ordinal_probs = ordinal_class_probs(outputs["ordinal_logits"])
    if prediction_mode == "ordinal":
        return ordinal_probs
    weight = min(1.0, max(0.0, float(ordinal_blend_weight)))
    probs = (1.0 - weight) * class_probs + weight * ordinal_probs
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def amp_context(device: torch.device, enabled: bool, bf16: bool) -> torch.autocast:
    device_type = "cuda" if device.type == "cuda" else "mps" if device.type == "mps" else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled and device.type != "cpu")


def output_score(outputs: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "direct":
        return outputs["direct_score"]
    if name == "hierarchical":
        return outputs["hierarchical_score"]
    if name == "covered_internal":
        return outputs["covered_internal_score"]
    if name == "legacy_ordinal":
        return outputs["legacy_ordinal_score"]
    if name == "noisy_or":
        return outputs["mil_noisy_or_score"]
    return outputs["score"]


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    ce_weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    covered_internal_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["label"].to(outputs["logits"].device)
    scores = batch["score"].to(outputs["logits"].device)
    sample_weight = batch["weight"].to(outputs["logits"].device)

    score_for_loss = output_score(outputs, args.score_loss_target)
    if args.score_loss == "mse":
        score_loss = F.mse_loss(score_for_loss, scores, reduction="none")
    else:
        score_loss = F.smooth_l1_loss(score_for_loss, scores, beta=args.huber_beta, reduction="none")
    score_loss = weighted_mean(score_loss, sample_weight)

    over = weighted_mean(F.relu(outputs["score"] - scores) ** 2, sample_weight)
    under = weighted_mean(F.relu(scores - outputs["score"]) ** 2, sample_weight)

    any_target = (labels >= 1).float()
    substantial_target = (labels >= 2).float()
    mostly_target = (labels >= 3).float()
    substantial_mask = labels >= 1
    mostly_mask = labels >= 2
    any_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["any_logits"].float(),
        any_target,
        pos_weight=torch.tensor(args.any_pos_weight, dtype=torch.float32, device=labels.device),
        reduction="none",
    )
    substantial_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["substantial_logits"].float(),
        substantial_target,
        pos_weight=torch.tensor(args.substantial_pos_weight, dtype=torch.float32, device=labels.device),
        reduction="none",
    )
    mostly_loss_raw = F.binary_cross_entropy_with_logits(
        outputs["mostly_logits"].float(),
        mostly_target,
        pos_weight=torch.tensor(args.mostly_pos_weight, dtype=torch.float32, device=labels.device),
        reduction="none",
    )
    any_loss = weighted_mean(any_loss_raw, sample_weight)
    substantial_loss = masked_weighted_mean(substantial_loss_raw, sample_weight, substantial_mask)
    mostly_loss = masked_weighted_mean(mostly_loss_raw, sample_weight, mostly_mask)
    hierarchical_loss = (
        args.lambda_any * any_loss
        + args.lambda_substantial * substantial_loss
        + args.lambda_mostly * mostly_loss
    )

    consistency = F.smooth_l1_loss(
        outputs["direct_score"],
        outputs["hierarchical_score"].detach(),
        beta=args.consistency_huber_beta,
        reduction="none",
    )
    consistency = weighted_mean(consistency, sample_weight)

    ce = F.cross_entropy(outputs["logits"], labels, weight=ce_weights, reduction="none")
    ce = weighted_mean(ce, sample_weight)
    covered_mask = labels >= 1
    if covered_mask.any():
        covered_internal_targets = (labels[covered_mask] - 1).clamp(min=0, max=len(COVERED_INTERNAL_LABELS) - 1)
        covered_internal_raw = F.cross_entropy(
            outputs["covered_internal_logits"][covered_mask],
            covered_internal_targets,
            weight=covered_internal_weights,
            reduction="none",
        )
        covered_internal = weighted_mean(covered_internal_raw, sample_weight[covered_mask])
    else:
        covered_internal = outputs["covered_internal_logits"].sum() * 0.0
    ordinal = F.binary_cross_entropy_with_logits(
        outputs["ordinal_logits"],
        ordinal_targets(labels),
        pos_weight=ordinal_weights,
        reduction="none",
    ).mean(dim=-1)
    ordinal = weighted_mean(ordinal, sample_weight)

    pair_mask = batch["pair_mask"].to(outputs["logits"].device)
    pair_target_mask = batch["pair_target_mask"].to(outputs["logits"].device) * pair_mask
    pair_targets = batch["pair_score_target"].to(outputs["logits"].device)
    pair_weights = pair_target_mask * sample_weight.unsqueeze(1)
    if args.score_loss == "mse":
        pair_loss_raw = F.mse_loss(outputs["pair_scores"], pair_targets, reduction="none")
    else:
        pair_loss_raw = F.smooth_l1_loss(
            outputs["pair_scores"],
            pair_targets,
            beta=args.huber_beta,
            reduction="none",
        )
    pair_aux = (pair_loss_raw * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)

    attention = outputs["attention"].clamp_min(1e-8)
    attention_entropy = -(attention * attention.log()).sum(dim=1).mean()

    total = (
        args.hierarchical_loss_weight * hierarchical_loss
        + args.score_loss_weight * score_loss
        + args.over_coverage_penalty * over
        + args.under_coverage_penalty * under
        + args.consistency_loss_weight * consistency
        + args.ce_loss_weight * ce
        + args.covered_internal_loss_weight * covered_internal
        + args.ordinal_loss_weight * ordinal
        + args.pair_aux_loss_weight * pair_aux
        + args.attention_entropy_weight * attention_entropy
    )
    return total, {
        "total": float(total.detach().cpu()),
        "score": float(score_loss.detach().cpu()),
        "over_penalty": float(over.detach().cpu()),
        "under_penalty": float(under.detach().cpu()),
        "hierarchical": float(hierarchical_loss.detach().cpu()),
        "any": float(any_loss.detach().cpu()),
        "substantial": float(substantial_loss.detach().cpu()),
        "mostly": float(mostly_loss.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "covered_internal": float(covered_internal.detach().cpu()),
        "ordinal": float(ordinal.detach().cpu()),
        "pair_aux": float(pair_aux.detach().cpu()),
        "attention_entropy": float(attention_entropy.detach().cpu()),
    }


def head_probability_diagnostics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    threshold = min(1.0, max(0.0, float(threshold)))
    metrics = binary_metrics(y_true, y_prob >= threshold, y_prob)
    metrics["valid_count"] = int(len(y_true))
    metrics["gold_positive_rate"] = round(float(y_true.astype(bool).mean()), 4) if len(y_true) else 0.0
    metrics["pred_positive_rate"] = round(float((y_prob >= threshold).mean()), 4) if len(y_prob) else 0.0
    metrics["mean_probability"] = round(float(y_prob.mean()), 4) if len(y_prob) else 0.0
    metrics["median_probability"] = round(float(np.median(y_prob)), 4) if len(y_prob) else 0.0
    metrics["threshold"] = round(float(threshold), 6)
    return metrics


def covered_internal_probability_diagnostics(
    y_true: np.ndarray,
    covered_internal_probs: np.ndarray,
) -> dict[str, Any]:
    covered_mask = y_true >= 1
    if not covered_mask.any():
        return {"valid_count": 0}
    covered_true = y_true[covered_mask] - 1
    covered_probs = covered_internal_probs[covered_mask]
    covered_pred = np.argmax(covered_probs, axis=1).astype(np.int64)
    per_class = {}
    for idx, label in enumerate(COVERED_INTERNAL_LABELS):
        tp = int(((covered_true == idx) & (covered_pred == idx)).sum())
        fp = int(((covered_true != idx) & (covered_pred == idx)).sum())
        fn = int(((covered_true == idx) & (covered_pred != idx)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int((covered_true == idx).sum()),
        }
    f1s = [item["f1"] for item in per_class.values()]
    metrics = {
        "accuracy": round(float((covered_true == covered_pred).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "per_class": per_class,
    }
    metrics["valid_count"] = int(covered_mask.sum())
    metrics["pred_distribution"] = dict(Counter(COVERED_INTERNAL_LABELS[int(idx)] for idx in covered_pred))
    metrics["true_distribution"] = dict(Counter(COVERED_INTERNAL_LABELS[int(idx)] for idx in covered_true))
    metrics["one_vs_rest"] = {
        label: head_probability_diagnostics(
            covered_true == idx,
            covered_probs[:, idx],
            1.0 / len(COVERED_INTERNAL_LABELS),
        )
        for idx, label in enumerate(COVERED_INTERNAL_LABELS)
    }
    return metrics


def disagreement_matrix(
    hierarchical_pred: np.ndarray,
    class_pred: np.ndarray,
    y_true: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for h_idx in range(len(JOINT_LABELS)):
        for c_idx in range(len(JOINT_LABELS)):
            mask = (hierarchical_pred == h_idx) & (class_pred == c_idx)
            count = int(mask.sum())
            if count == 0:
                continue
            gold = y_true[mask]
            rows.append(
                {
                    "hierarchical": JOINT_LABELS[h_idx],
                    "class": JOINT_LABELS[c_idx],
                    "count": count,
                    "gold_partial_rate": round(float((gold == 2).mean()), 4),
                    "gold_mostly_rate": round(float((gold == 3).mean()), 4),
                    "gold_distribution": dict(Counter(JOINT_LABELS[int(idx)] for idx in gold)),
                }
            )
    rows.sort(key=lambda row: (-int(row["count"]), row["hierarchical"], row["class"]))
    return rows


def selection_constraints_satisfied(metrics: dict[str, Any], label_key: str = "conditional_label_metrics") -> bool:
    label = metrics[label_key]
    return (
        float(label["not_covered_recall"]) >= 0.70
        and float(label["weak_recall"]) >= 0.10
        and float(label["partial_recall"]) >= 0.20
        and float(label["mostly_recall"]) >= 0.35
        and float(label["covered_binary"]["recall"]) >= 0.70
        and distribution_rate(label, "pred_distribution", "partially_covered") <= 0.25
        and distribution_rate(label, "pred_distribution", "mostly_covered") <= 0.30
    )


def metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "neg_score_mae":
        return -float(metrics["score_metrics"]["mae"])
    if name == "score_pearson":
        return float(metrics["score_metrics"]["pearson"])
    if name == "score_spearman":
        return float(metrics["score_metrics"]["spearman"])
    if name == "joint_macro_f1":
        return float(metrics["hierarchical_label_metrics"]["macro_f1"])
    if name == "joint_qwk":
        return float(metrics["hierarchical_label_metrics"]["qwk"])
    if name == "covered_recall":
        return float(metrics["hierarchical_label_metrics"]["covered_binary"]["recall"])
    if name == "conditional_macro_f1":
        return float(metrics["conditional_label_metrics"]["macro_f1"])
    if name == "conditional_qwk":
        return float(metrics["conditional_label_metrics"]["qwk"])
    if name == "covered_internal_macro_f1":
        return float(metrics["covered_internal_label_metrics"]["macro_f1"])
    if name == "covered_internal_qwk":
        return float(metrics["covered_internal_label_metrics"]["qwk"])
    if name == "internal_blend_macro_f1":
        return float(metrics["internal_blend_label_metrics"]["macro_f1"])
    if name == "internal_blend_qwk":
        return float(metrics["internal_blend_label_metrics"]["qwk"])
    if name == "hier_class_macro_f1":
        return float(metrics["hier_class_blend_label_metrics"]["macro_f1"])
    if name == "hier_class_qwk":
        return float(metrics["hier_class_blend_label_metrics"]["qwk"])
    if name == "constrained_balanced_score":
        base = (
            float(metrics["conditional_label_metrics"]["macro_f1"])
            + 0.25 * float(metrics["conditional_label_metrics"]["qwk"])
            + 0.10 * float(metrics["score_metrics"]["spearman"])
            - 0.25 * float(metrics["score_metrics"]["mae"])
        )
        if selection_constraints_satisfied(metrics):
            return base
        label = metrics["conditional_label_metrics"]
        deficits = [
            max(0.0, 0.70 - float(label["not_covered_recall"])),
            max(0.0, 0.10 - float(label["weak_recall"])),
            max(0.0, 0.20 - float(label["partial_recall"])),
            max(0.0, 0.35 - float(label["mostly_recall"])),
            max(0.0, 0.70 - float(label["covered_binary"]["recall"])),
            max(0.0, distribution_rate(label, "pred_distribution", "partially_covered") - 0.25),
            max(0.0, distribution_rate(label, "pred_distribution", "mostly_covered") - 0.30),
        ]
        return -1.0 - sum(deficits) + 0.01 * base
    if name == "selected_constrained_balanced_score":
        label = metrics["selected_head_label_metrics"]
        base = (
            float(label["macro_f1"])
            + 0.25 * float(label["qwk"])
            + 0.10 * float(metrics["score_metrics"]["spearman"])
            - 0.25 * float(metrics["score_metrics"]["mae"])
        )
        if selection_constraints_satisfied(metrics, "selected_head_label_metrics"):
            return base
        deficits = [
            max(0.0, 0.70 - float(label["not_covered_recall"])),
            max(0.0, 0.10 - float(label["weak_recall"])),
            max(0.0, 0.20 - float(label["partial_recall"])),
            max(0.0, 0.35 - float(label["mostly_recall"])),
            max(0.0, 0.70 - float(label["covered_binary"]["recall"])),
            max(0.0, distribution_rate(label, "pred_distribution", "partially_covered") - 0.25),
            max(0.0, distribution_rate(label, "pred_distribution", "mostly_covered") - 0.30),
        ]
        return -1.0 - sum(deficits) + 0.01 * base
    if name == "balanced_score":
        return (
            float(metrics["score_metrics"]["pearson"])
            + 0.20 * float(metrics["conditional_label_metrics"]["qwk"])
            + 0.15 * float(metrics["hier_class_blend_label_metrics"]["qwk"])
            + 0.15 * float(metrics["conditional_label_metrics"]["covered_binary"]["recall"])
            - 0.50 * float(metrics["score_metrics"]["mean_over_error"])
            - 0.25 * float(metrics["score_metrics"]["mae"])
        )
    return float(metrics["score_decoded_label_metrics"][name])


@torch.no_grad()
def evaluate(
    model: JointMilModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    ce_weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    covered_internal_weights: torch.Tensor,
    thresholds: tuple[float, float, float],
    conditional_thresholds: tuple[float, float, float],
    args: argparse.Namespace,
    return_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    y_true_label: list[int] = []
    y_score_true: list[float] = []
    y_score_pred: list[float] = []
    y_direct_score_pred: list[float] = []
    y_hierarchical_score_pred: list[float] = []
    y_covered_internal_score_pred: list[float] = []
    y_legacy_ordinal_score_pred: list[float] = []
    y_noisy_or_pred: list[float] = []
    selected_argmax: list[int] = []
    hierarchical_argmax: list[int] = []
    conditional_pred_all: list[int] = []
    covered_internal_pred_all: list[int] = []
    internal_blend_argmax: list[int] = []
    hier_class_blend_argmax: list[int] = []
    class_argmax: list[int] = []
    legacy_ordinal_argmax: list[int] = []
    any_prob_all: list[float] = []
    substantial_prob_all: list[float] = []
    mostly_prob_all: list[float] = []
    covered_internal_prob_all: list[list[float]] = []
    prob_all: list[list[float]] = []
    max_attention: list[float] = []
    graph_feature_gates: list[float] = []
    losses: dict[str, list[float]] = defaultdict(list)
    pred_rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        inputs, pair_mask, graph_features = move_inputs_to_device(batch, device)
        with amp_context(device, args.amp, args.bf16):
            outputs = model(inputs, pair_mask, graph_features)
            loss, parts = compute_loss(outputs, batch, ce_weights, ordinal_weights, covered_internal_weights, args)
        probs = probabilities(
            outputs,
            args.prediction_mode,
            args.ordinal_blend_weight,
            args.hier_class_blend_weight,
            args.internal_blend_weight,
        )
        selected_pred = probs.argmax(dim=-1).detach().cpu().numpy()
        class_probs = torch.softmax(outputs["logits"].float(), dim=-1)
        class_pred = class_probs.argmax(dim=-1).detach().cpu().numpy()
        hierarchical_probs = outputs["hierarchical_probs"].float()
        hierarchical_pred = hierarchical_probs.argmax(dim=-1).detach().cpu().numpy()
        hier_class_blend_probs = probabilities(
            outputs,
            "hier_class_blend",
            args.ordinal_blend_weight,
            args.hier_class_blend_weight,
            args.internal_blend_weight,
        )
        hier_class_blend_pred = hier_class_blend_probs.argmax(dim=-1).detach().cpu().numpy()
        internal_blend_probs = probabilities(
            outputs,
            "internal_blend",
            args.ordinal_blend_weight,
            args.hier_class_blend_weight,
            args.internal_blend_weight,
        )
        internal_blend_pred = internal_blend_probs.argmax(dim=-1).detach().cpu().numpy()
        legacy_ordinal_probs = outputs["legacy_ordinal_probs"].float()
        legacy_ordinal_pred = legacy_ordinal_probs.argmax(dim=-1).detach().cpu().numpy()
        pred_scores = outputs["score"].detach().float().cpu().numpy()
        direct_scores = outputs["direct_score"].detach().float().cpu().numpy()
        hierarchical_scores = outputs["hierarchical_score"].detach().float().cpu().numpy()
        covered_internal_scores = outputs["covered_internal_score"].detach().float().cpu().numpy()
        legacy_ordinal_scores = outputs["legacy_ordinal_score"].detach().float().cpu().numpy()
        noisy_scores = outputs["mil_noisy_or_score"].detach().float().cpu().numpy()
        any_probs = torch.sigmoid(outputs["any_logits"].float()).detach().cpu().numpy()
        substantial_probs = torch.sigmoid(outputs["substantial_logits"].float()).detach().cpu().numpy()
        mostly_probs = torch.sigmoid(outputs["mostly_logits"].float()).detach().cpu().numpy()
        covered_internal_probs = torch.softmax(outputs["covered_internal_logits"].float(), dim=-1).detach().cpu().numpy()
        conditional_pred = conditional_label_ids(any_probs, substantial_probs, mostly_probs, conditional_thresholds)
        covered_internal_pred = covered_internal_label_ids(
            any_probs,
            covered_internal_probs,
            args.covered_internal_any_threshold,
        )
        if args.prediction_mode == "conditional":
            selected_pred = conditional_pred
        elif args.prediction_mode == "covered_internal":
            selected_pred = covered_internal_pred
        labels = batch["label"].numpy()
        scores = batch["score"].numpy()
        attention = outputs["attention"].detach().float().cpu().numpy()
        pair_scores = outputs["pair_scores"].detach().float().cpu().numpy()
        graph_feature_gate = outputs["graph_feature_gate"].detach().float().cpu().numpy()

        y_true_label.extend(int(x) for x in labels)
        y_score_true.extend(float(x) for x in scores)
        y_score_pred.extend(float(x) for x in pred_scores)
        y_direct_score_pred.extend(float(x) for x in direct_scores)
        y_hierarchical_score_pred.extend(float(x) for x in hierarchical_scores)
        y_covered_internal_score_pred.extend(float(x) for x in covered_internal_scores)
        y_legacy_ordinal_score_pred.extend(float(x) for x in legacy_ordinal_scores)
        y_noisy_or_pred.extend(float(x) for x in noisy_scores)
        selected_argmax.extend(int(x) for x in selected_pred)
        hierarchical_argmax.extend(int(x) for x in hierarchical_pred)
        conditional_pred_all.extend(int(x) for x in conditional_pred)
        covered_internal_pred_all.extend(int(x) for x in covered_internal_pred)
        internal_blend_argmax.extend(int(x) for x in internal_blend_pred)
        hier_class_blend_argmax.extend(int(x) for x in hier_class_blend_pred)
        class_argmax.extend(int(x) for x in class_pred)
        legacy_ordinal_argmax.extend(int(x) for x in legacy_ordinal_pred)
        any_prob_all.extend(float(x) for x in any_probs)
        substantial_prob_all.extend(float(x) for x in substantial_probs)
        mostly_prob_all.extend(float(x) for x in mostly_probs)
        covered_internal_prob_all.extend(covered_internal_probs.astype(np.float32).tolist())
        max_attention.extend(float(x) for x in attention.max(axis=1).tolist())
        graph_feature_gates.extend(float(x) for x in graph_feature_gate.tolist())
        prob_batch = probs.detach().float().cpu().numpy().tolist()
        class_prob_batch = class_probs.detach().float().cpu().numpy().tolist()
        hierarchy_prob_batch = hierarchical_probs.detach().float().cpu().numpy().tolist()
        hier_class_blend_prob_batch = hier_class_blend_probs.detach().float().cpu().numpy().tolist()
        covered_internal_joint_prob_batch = outputs["covered_internal_probs"].detach().float().cpu().numpy().tolist()
        covered_internal_prob_batch = covered_internal_probs.astype(np.float32).tolist()
        internal_blend_prob_batch = internal_blend_probs.detach().float().cpu().numpy().tolist()
        legacy_ordinal_prob_batch = legacy_ordinal_probs.detach().float().cpu().numpy().tolist()
        prob_all.extend(prob_batch)
        for name, value in parts.items():
            losses[name].append(value)
        losses["loss"].append(float(loss.detach().cpu()))

        if return_predictions:
            score_label_ids = score_to_label_ids(pred_scores, thresholds)
            for idx, target_id in enumerate(batch["target_idea_id"]):
                prompt_priors = []
                for prior_idx, item in enumerate(batch["prompt_priors"][idx]):
                    enriched = dict(item)
                    enriched["attention"] = round(float(attention[idx][prior_idx]), 6)
                    enriched["pred_pair_evidence_score"] = round(float(pair_scores[idx][prior_idx]), 6)
                    prompt_priors.append(enriched)
                pred_rows.append(
                    {
                        "target_idea_id": target_id,
                        "gold_joint_coverage_label": JOINT_LABELS[int(labels[idx])],
                        "gold_joint_coverage_score": round(float(scores[idx]), 6),
                        "pred_joint_coverage_label": JOINT_LABELS[int(score_label_ids[idx])],
                        "pred_joint_coverage_score": round(float(pred_scores[idx]), 6),
                        "pred_direct_joint_coverage_score": round(float(direct_scores[idx]), 6),
                        "pred_hierarchical_joint_coverage_score": round(float(hierarchical_scores[idx]), 6),
                        "pred_covered_internal_joint_coverage_score": round(float(covered_internal_scores[idx]), 6),
                        "pred_legacy_ordinal_joint_coverage_score": round(float(legacy_ordinal_scores[idx]), 6),
                        "class_head_pred_joint_coverage_label": JOINT_LABELS[int(class_pred[idx])],
                        "hierarchical_head_pred_joint_coverage_label": JOINT_LABELS[int(hierarchical_pred[idx])],
                        "conditional_head_pred_joint_coverage_label": JOINT_LABELS[int(conditional_pred[idx])],
                        "covered_internal_head_pred_joint_coverage_label": JOINT_LABELS[int(covered_internal_pred[idx])],
                        "internal_blend_pred_joint_coverage_label": JOINT_LABELS[int(internal_blend_pred[idx])],
                        "hier_class_blend_pred_joint_coverage_label": JOINT_LABELS[int(hier_class_blend_pred[idx])],
                        "legacy_ordinal_head_pred_joint_coverage_label": JOINT_LABELS[int(legacy_ordinal_pred[idx])],
                        "selected_head_pred_joint_coverage_label": JOINT_LABELS[int(selected_pred[idx])],
                        "pred_any_coverage_probability": round(float(any_probs[idx]), 6),
                        "pred_substantial_probability_given_covered": round(float(substantial_probs[idx]), 6),
                        "pred_mostly_probability_given_substantial": round(float(mostly_probs[idx]), 6),
                        "mil_noisy_or_score": round(float(noisy_scores[idx]), 6),
                        "graph_feature_gate": round(float(graph_feature_gate[idx]), 6),
                        "pred_joint_probabilities": {
                            label: round(float(prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_hierarchical_joint_probabilities": {
                            label: round(float(hierarchy_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_class_head_joint_probabilities": {
                            label: round(float(class_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_covered_internal_joint_probabilities": {
                            label: round(float(covered_internal_joint_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_covered_internal_conditional_probabilities": {
                            label: round(float(covered_internal_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(COVERED_INTERNAL_LABELS)
                        },
                        "pred_internal_blend_joint_probabilities": {
                            label: round(float(internal_blend_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_hier_class_blend_joint_probabilities": {
                            label: round(float(hier_class_blend_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "pred_legacy_ordinal_joint_probabilities": {
                            label: round(float(legacy_ordinal_prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "ranked_priors": batch["ranking"][idx].get("ranked_priors")
                        or batch["ranking"][idx].get("candidate_priors"),
                        "prompt_priors": prompt_priors,
                    }
                )

    true_label_np = np.asarray(y_true_label, dtype=np.int64)
    score_true_np = np.asarray(y_score_true, dtype=np.float32)
    score_pred_np = np.asarray(y_score_pred, dtype=np.float32)
    direct_score_np = np.asarray(y_direct_score_pred, dtype=np.float32)
    hierarchical_score_np = np.asarray(y_hierarchical_score_pred, dtype=np.float32)
    covered_internal_score_np = np.asarray(y_covered_internal_score_pred, dtype=np.float32)
    legacy_ordinal_score_np = np.asarray(y_legacy_ordinal_score_pred, dtype=np.float32)
    noisy_score_np = np.asarray(y_noisy_or_pred, dtype=np.float32)
    score_label_np = score_to_label_ids(score_pred_np, thresholds)
    selected_argmax_np = np.asarray(selected_argmax, dtype=np.int64)
    hierarchical_argmax_np = np.asarray(hierarchical_argmax, dtype=np.int64)
    conditional_pred_np = np.asarray(conditional_pred_all, dtype=np.int64)
    covered_internal_pred_np = np.asarray(covered_internal_pred_all, dtype=np.int64)
    internal_blend_argmax_np = np.asarray(internal_blend_argmax, dtype=np.int64)
    hier_class_blend_argmax_np = np.asarray(hier_class_blend_argmax, dtype=np.int64)
    class_argmax_np = np.asarray(class_argmax, dtype=np.int64)
    legacy_ordinal_argmax_np = np.asarray(legacy_ordinal_argmax, dtype=np.int64)
    any_prob_np = np.asarray(any_prob_all, dtype=np.float32)
    substantial_prob_np = np.asarray(substantial_prob_all, dtype=np.float32)
    mostly_prob_np = np.asarray(mostly_prob_all, dtype=np.float32)
    covered_internal_prob_np = np.asarray(covered_internal_prob_all, dtype=np.float32)
    covered_mask = true_label_np >= 1
    substantial_mask = true_label_np >= 2

    metrics = {
        "score_metrics": score_metrics(score_true_np, score_pred_np),
        "direct_score_metrics": score_metrics(score_true_np, direct_score_np),
        "hierarchical_score_metrics": score_metrics(score_true_np, hierarchical_score_np),
        "covered_internal_score_metrics": score_metrics(score_true_np, covered_internal_score_np),
        "legacy_ordinal_score_metrics": score_metrics(score_true_np, legacy_ordinal_score_np),
        "mil_noisy_or_score_metrics": score_metrics(score_true_np, noisy_score_np),
        "score_distribution_by_true_label": {
            "final": score_distribution_by_label(true_label_np, score_pred_np),
            "direct": score_distribution_by_label(true_label_np, direct_score_np),
            "hierarchical": score_distribution_by_label(true_label_np, hierarchical_score_np),
            "covered_internal": score_distribution_by_label(true_label_np, covered_internal_score_np),
            "legacy_ordinal": score_distribution_by_label(true_label_np, legacy_ordinal_score_np),
            "noisy_or": score_distribution_by_label(true_label_np, noisy_score_np),
        },
        "score_decoded_label_metrics": enriched_label_metrics(true_label_np, score_label_np),
        "selected_head_label_metrics": enriched_label_metrics(true_label_np, selected_argmax_np),
        "hierarchical_label_metrics": enriched_label_metrics(true_label_np, hierarchical_argmax_np),
        "conditional_label_metrics": enriched_label_metrics(true_label_np, conditional_pred_np),
        "covered_internal_label_metrics": enriched_label_metrics(true_label_np, covered_internal_pred_np),
        "internal_blend_label_metrics": enriched_label_metrics(true_label_np, internal_blend_argmax_np),
        "hier_class_blend_label_metrics": enriched_label_metrics(true_label_np, hier_class_blend_argmax_np),
        "class_head_label_metrics": enriched_label_metrics(true_label_np, class_argmax_np),
        "legacy_ordinal_label_metrics": enriched_label_metrics(true_label_np, legacy_ordinal_argmax_np),
        "hierarchical_head_metrics": {
            "any_coverage": head_probability_diagnostics(true_label_np >= 1, any_prob_np, 0.5),
            "substantial_given_covered": head_probability_diagnostics(
                true_label_np[covered_mask] >= 2,
                substantial_prob_np[covered_mask],
                0.5,
            ),
            "mostly_given_substantial": head_probability_diagnostics(
                true_label_np[substantial_mask] >= 3,
                mostly_prob_np[substantial_mask],
                0.5,
            ),
        },
        "conditional_threshold_head_metrics": {
            "any_coverage": head_probability_diagnostics(true_label_np >= 1, any_prob_np, conditional_thresholds[0]),
            "substantial_given_covered": head_probability_diagnostics(
                true_label_np[covered_mask] >= 2,
                substantial_prob_np[covered_mask],
                conditional_thresholds[1],
            ),
            "mostly_given_substantial": head_probability_diagnostics(
                true_label_np[substantial_mask] >= 3,
                mostly_prob_np[substantial_mask],
                conditional_thresholds[2],
            ),
        },
        "covered_internal_head_metrics": {
            "any_coverage": head_probability_diagnostics(
                true_label_np >= 1,
                any_prob_np,
                args.covered_internal_any_threshold,
            ),
            "covered_internal_given_covered": covered_internal_probability_diagnostics(
                true_label_np,
                covered_internal_prob_np,
            ),
        },
        "hierarchical_class_disagreement_matrix": disagreement_matrix(
            hierarchical_argmax_np,
            class_argmax_np,
            true_label_np,
        ),
        "attention_summary": {
            "mean_max_attention": round(float(np.mean(max_attention)), 4) if max_attention else 0.0,
            "median_max_attention": round(float(np.median(max_attention)), 4) if max_attention else 0.0,
        },
        "graph_feature_summary": {
            "enabled": bool(graph_feature_gates and max(graph_feature_gates) > 0),
            "mean_gate": round(float(np.mean(graph_feature_gates)), 6) if graph_feature_gates else 0.0,
            "median_gate": round(float(np.median(graph_feature_gates)), 6) if graph_feature_gates else 0.0,
        },
        "loss": {name: round(float(np.mean(values)), 6) for name, values in losses.items()},
    }
    if return_predictions:
        metrics["_predictions"] = pred_rows
    return metrics


def sampler_weights(dataset: JointMilDataset, strategy: str) -> torch.Tensor | None:
    if strategy == "none":
        return None
    counts = Counter(int(item["label"]) for item in dataset.items)
    alpha = 0.5 if strategy == "sqrt" else 1.0
    weights = [
        float(item["weight"]) / max(1.0, float(counts[int(item["label"])]) ** alpha)
        for item in dataset.items
    ]
    return torch.tensor(weights, dtype=torch.double)


def expected_sampler_distribution(dataset: JointMilDataset, strategy: str) -> dict[str, Any]:
    counts = Counter(int(item["label"]) for item in dataset.items)
    if strategy == "none":
        total = sum(counts.values())
        return {
            JOINT_LABELS[idx]: {
                "count": int(counts.get(idx, 0)),
                "expected_rate": round(float(counts.get(idx, 0) / max(1, total)), 4),
            }
            for idx in range(len(JOINT_LABELS))
        }
    weights = sampler_weights(dataset, strategy)
    if weights is None:
        return {}
    weight_np = weights.detach().cpu().numpy().astype(np.float64)
    labels = np.asarray([int(item["label"]) for item in dataset.items], dtype=np.int64)
    total_weight = float(weight_np.sum())
    out = {}
    for idx, label in enumerate(JOINT_LABELS):
        class_weight = float(weight_np[labels == idx].sum())
        out[label] = {
            "count": int(counts.get(idx, 0)),
            "expected_sampled_count": round(float(len(dataset) * class_weight / max(1e-12, total_weight)), 2),
            "expected_rate": round(float(class_weight / max(1e-12, total_weight)), 4),
        }
    return out


def ce_weight_tensor(args: argparse.Namespace, dataset: JointMilDataset, device: torch.device) -> torch.Tensor:
    if args.ce_class_weight_mode == "none":
        weights = [1.0 for _ in JOINT_LABELS]
    elif args.ce_class_weight_mode == "mild_partial":
        weights = [1.0, 1.0, float(args.partial_ce_weight), 1.0]
    else:
        return class_weights(dataset, device)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def covered_internal_weight_tensor(args: argparse.Namespace, dataset: JointMilDataset, device: torch.device) -> torch.Tensor:
    if args.covered_internal_class_weight_mode == "none":
        weights = [1.0 for _ in COVERED_INTERNAL_LABELS]
        return torch.tensor(weights, dtype=torch.float32, device=device)
    counts = Counter(int(item["label"]) - 1 for item in dataset.items if int(item["label"]) >= 1)
    total = sum(counts.values())
    if total <= 0:
        weights = [1.0 for _ in COVERED_INTERNAL_LABELS]
    else:
        alpha = 0.5 if args.covered_internal_class_weight_mode == "sqrt" else 1.0
        weights = [
            (total / max(1.0, float(len(COVERED_INTERNAL_LABELS) * counts.get(idx, 0)))) ** alpha
            for idx in range(len(COVERED_INTERNAL_LABELS))
        ]
        mean_weight = float(np.mean(weights)) if weights else 1.0
        weights = [min(float(args.covered_internal_max_class_weight), weight / max(1e-8, mean_weight)) for weight in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loader(
    dataset: JointMilDataset,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    balanced_sampling: str = "none",
) -> DataLoader[dict[str, Any]]:
    sampler = None
    if shuffle:
        weights = sampler_weights(dataset, balanced_sampling)
        if weights is not None:
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=JointMilCollator(tokenizer, max_length, dataset.top_k),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_tokenizer(args: argparse.Namespace) -> Any:
    kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    try:
        return AutoTokenizer.from_pretrained(args.model_name, use_fast=True, **kwargs)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "warning": "fast_tokenizer_load_failed_using_slow",
                    "model_name": args.model_name,
                    "error": str(exc).splitlines()[0],
                },
                ensure_ascii=False,
            )
        )
        return AutoTokenizer.from_pretrained(args.model_name, use_fast=False, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-dir", type=Path, default=None)
    parser.add_argument("--ranking-file", type=Path, default=None)
    parser.add_argument("--train-ranking-file", type=Path, default=None)
    parser.add_argument("--dev-ranking-file", type=Path, default=None)
    parser.add_argument("--test-ranking-file", type=Path, default=None)
    parser.add_argument("--novelty-file", type=Path, default=Path("dataset/idea_novelty_dataset.jsonl"))
    parser.add_argument("--pair-text-file", type=Path, action="append", default=[])
    parser.add_argument("--graph-dir", type=Path, default=Path("pipeline/output/graph_full_xtype"))
    parser.add_argument("--canonical-ideas", type=Path, default=None)
    parser.add_argument("--graph-file", type=Path, default=None)
    parser.add_argument("--include-graph-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-direct-target-prior-graph-relations", action="store_true")
    parser.add_argument("--graph-context-relations", default=",".join(GRAPH_RELATIONS))
    parser.add_argument("--graph-feature-max-one-hop", type=int, default=32)
    parser.add_argument("--graph-feature-max-two-hop-per-neighbor", type=int, default=16)
    parser.add_argument("--graph-feature-gate-init", type=float, default=-3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/stage_joint_coverage_mil_text_deberta"))
    parser.add_argument(
        "--eval-only-checkpoint",
        type=Path,
        default=None,
        help=(
            "Skip training, load this Joint checkpoint, retune configured dev thresholds, "
            "and export complete non-sampled train/dev/test predictions."
        ),
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--noisy-or-blend-weight", type=float, default=0.25)
    parser.add_argument(
        "--score-prediction-mode",
        choices=["hierarchical", "covered_internal", "direct", "blend", "noisy_or", "noisy_blend"],
        default="hierarchical",
    )
    parser.add_argument("--score-blend-weight", type=float, default=0.30, help="Direct-score weight when --score-prediction-mode blend.")
    parser.add_argument("--score-loss", choices=["huber", "mse"], default="mse")
    parser.add_argument("--score-loss-target", choices=["direct", "hierarchical", "covered_internal", "legacy_ordinal", "noisy_or", "final"], default="hierarchical")
    parser.add_argument("--huber-beta", type=float, default=0.20)
    parser.add_argument("--score-loss-weight", type=float, default=0.30)
    parser.add_argument("--over-coverage-penalty", type=float, default=0.0)
    parser.add_argument("--under-coverage-penalty", type=float, default=0.0)
    parser.add_argument("--hierarchical-loss-weight", type=float, default=1.0)
    parser.add_argument("--any-pos-weight", type=float, default=1.30)
    parser.add_argument("--substantial-pos-weight", type=float, default=1.0)
    parser.add_argument("--mostly-pos-weight", type=float, default=0.70)
    parser.add_argument("--lambda-any", type=float, default=1.0)
    parser.add_argument("--lambda-substantial", type=float, default=1.0)
    parser.add_argument("--lambda-mostly", type=float, default=0.75)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--consistency-huber-beta", type=float, default=0.10)
    parser.add_argument("--ce-loss-weight", type=float, default=0.15)
    parser.add_argument("--ce-class-weight-mode", choices=["mild_partial", "auto", "none"], default="mild_partial")
    parser.add_argument("--partial-ce-weight", type=float, default=1.20)
    parser.add_argument("--covered-internal-loss-weight", type=float, default=0.35)
    parser.add_argument("--covered-internal-class-weight-mode", choices=["sqrt", "auto", "none"], default="sqrt")
    parser.add_argument("--covered-internal-max-class-weight", type=float, default=3.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--pair-aux-loss-weight", type=float, default=0.15)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.0)
    parser.add_argument(
        "--pair-aux-target",
        choices=["gold", "gold_effective", "pred", "none"],
        default="gold",
        help=(
            "Pair-score supervision inside Joint MIL. gold_effective applies the exact "
            "label-aware contribution mapping used to build V5 Joint Coverage targets."
        ),
    )
    parser.add_argument("--include-pred-pair-signals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gold-pair-signals", action="store_true")
    parser.add_argument(
        "--prediction-mode",
        choices=[
            "hierarchical",
            "separate",
            "conditional",
            "covered_internal",
            "internal_blend",
            "hier_class_blend",
            "class",
            "ordinal",
            "blend",
        ],
        default="conditional",
    )
    parser.add_argument("--ordinal-blend-weight", type=float, default=0.25)
    parser.add_argument("--hier-class-blend-weight", type=float, default=0.30)
    parser.add_argument("--internal-blend-weight", type=float, default=0.50)
    parser.add_argument("--label-thresholds", default="0.25,0.50,0.78")
    parser.add_argument("--conditional-thresholds", default="0.50,0.58,0.53")
    parser.add_argument("--covered-internal-any-threshold", type=float, default=0.50)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "balanced_score",
            "joint_macro_f1",
            "joint_qwk",
            "covered_recall",
            "conditional_macro_f1",
            "conditional_qwk",
            "covered_internal_macro_f1",
            "covered_internal_qwk",
            "internal_blend_macro_f1",
            "internal_blend_qwk",
            "constrained_balanced_score",
            "selected_constrained_balanced_score",
            "hier_class_macro_f1",
            "hier_class_qwk",
            "neg_score_mae",
            "score_pearson",
            "score_spearman",
            "qwk",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
        ],
        default="constrained_balanced_score",
    )
    parser.add_argument("--balanced-sampling", choices=["none", "sqrt", "label"], default="sqrt")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--tune-thresholds-on-dev", action="store_true")
    parser.add_argument("--tune-conditional-thresholds-on-dev", action="store_true")
    parser.add_argument("--tune-covered-internal-threshold-on-dev", action="store_true")
    parser.add_argument("--tune-internal-blend-on-dev", action="store_true")
    parser.add_argument("--tune-hier-class-blend-on-dev", action="store_true")
    parser.add_argument("--threshold-search-metric", choices=["balanced", "macro_f1", "qwk", "balanced_accuracy", "neg_ordinal_mae"], default="balanced")
    parser.add_argument("--threshold-steps", type=int, default=41)
    parser.add_argument("--conditional-any-range", default="0.30,0.70")
    parser.add_argument("--conditional-substantial-range", default="0.30,0.70")
    parser.add_argument("--conditional-mostly-range", default="0.50,0.62")
    parser.add_argument("--covered-internal-any-range", default="0.30,0.70")
    parser.add_argument("--threshold-min-not-recall", type=float, default=0.70)
    parser.add_argument("--threshold-min-weak-recall", type=float, default=0.10)
    parser.add_argument("--threshold-min-partial-recall", type=float, default=0.20)
    parser.add_argument("--threshold-min-mostly-recall", type=float, default=0.35)
    parser.add_argument("--threshold-min-covered-recall", type=float, default=0.70)
    parser.add_argument("--threshold-max-partial-pred-ratio", type=float, default=0.22)
    parser.add_argument("--threshold-max-mostly-pred-ratio", type=float, default=0.30)
    parser.add_argument("--threshold-max-partial-pred-multiplier", type=float, default=3.0)
    parser.add_argument("--threshold-max-mostly-pred-multiplier", type=float, default=2.5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=257)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if (
        args.eval_only_checkpoint is not None
        and args.output_dir.resolve() == args.eval_only_checkpoint.resolve().parent.parent
    ):
        parser.error("--eval-only-checkpoint requires a fresh --output-dir; in-place export is not allowed")

    thresholds = parse_thresholds(args.label_thresholds)
    conditional_thresholds = parse_conditional_thresholds(args.conditional_thresholds)
    conditional_any_range = parse_threshold_range(args.conditional_any_range, "conditional-any-range")
    conditional_substantial_range = parse_threshold_range(
        args.conditional_substantial_range,
        "conditional-substantial-range",
    )
    conditional_mostly_range = parse_threshold_range(args.conditional_mostly_range, "conditional-mostly-range")
    covered_internal_any_range = parse_threshold_range(
        args.covered_internal_any_range,
        "covered-internal-any-range",
    )
    args.partial_ce_weight = min(5.0, max(0.1, float(args.partial_ce_weight)))
    args.covered_internal_loss_weight = min(5.0, max(0.0, float(args.covered_internal_loss_weight)))
    args.covered_internal_max_class_weight = min(20.0, max(1.0, float(args.covered_internal_max_class_weight)))
    args.covered_internal_any_threshold = min(1.0, max(0.0, float(args.covered_internal_any_threshold)))
    args.internal_blend_weight = min(1.0, max(0.0, float(args.internal_blend_weight)))
    set_seed(args.seed)
    novelty_index = load_novelty_index(args.novelty_file)
    text_lookup = build_text_lookup(novelty_index, args.pair_text_file)
    split_rows, split_source = load_ranking_splits(args, novelty_index)
    if args.max_train_samples:
        split_rows["train"] = sample_train_rows(split_rows["train"], novelty_index, args.max_train_samples, args.seed)
    if args.max_eval_samples:
        split_rows["dev"] = split_rows["dev"][: args.max_eval_samples]
        split_rows["test"] = split_rows["test"][: args.max_eval_samples]

    graph_context: GraphStructureContext | None = None
    graph_canonical_ideas = args.canonical_ideas or (args.graph_dir / "canonical_ideas.json")
    graph_file = args.graph_file or (args.graph_dir / "graph.json")
    graph_relations = {item.strip() for item in str(args.graph_context_relations).split(",") if item.strip()}
    if args.include_graph_features:
        graph_context = GraphStructureContext(
            canonical_ideas=graph_canonical_ideas,
            graph_file=graph_file,
            relations=graph_relations,
        )

    datasets = {
        split: JointMilDataset(
            rows,
            novelty_index,
            text_lookup,
            args.top_k,
            args.include_pred_pair_signals,
            args.include_gold_pair_signals,
            args.pair_aux_target,
            graph_context=graph_context,
            graph_feature_max_one_hop=args.graph_feature_max_one_hop,
            graph_feature_max_two_hop_per_neighbor=args.graph_feature_max_two_hop_per_neighbor,
            include_direct_target_prior_graph_relations=args.include_direct_target_prior_graph_relations,
        )
        for split, rows in split_rows.items()
    }

    device = device_from_arg(args.device)
    args.gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    class_score_means = class_score_means_from_dataset(datasets["train"])
    tokenizer = load_tokenizer(args)
    cfg = JointMilConfig(
        model_name=args.model_name,
        dropout=args.dropout,
        noisy_or_blend_weight=args.noisy_or_blend_weight,
        score_prediction_mode=args.score_prediction_mode,
        score_blend_weight=args.score_blend_weight,
        graph_feature_dim=len(graph_context.feature_names) if graph_context else 0,
        graph_feature_gate_init=args.graph_feature_gate_init,
        class_score_means=class_score_means,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model = JointMilModel(cfg).float().to(device)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    loaders = {
        "train": build_loader(
            datasets["train"],
            tokenizer,
            args.batch_size,
            args.max_length,
            True,
            args.num_workers,
            args.balanced_sampling,
        ),
        "dev": build_loader(datasets["dev"], tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers),
        "test": build_loader(datasets["test"], tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers),
    }
    final_loaders = {
        "train": build_loader(
            datasets["train"],
            tokenizer,
            args.eval_batch_size,
            args.max_length,
            False,
            args.num_workers,
        ),
        "dev": loaders["dev"],
        "test": loaders["test"],
    }
    ce_weights = ce_weight_tensor(args, datasets["train"], device)
    ordinal_weights = ordinal_pos_weights(datasets["train"], device)
    covered_internal_weights = covered_internal_weight_tensor(args, datasets["train"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(0.01, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": {
            key: [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "model_config": asdict(cfg),
        "labels": JOINT_LABELS,
        "label_thresholds": {
            "weak": thresholds[0],
            "partial": thresholds[1],
            "mostly": thresholds[2],
        },
        "conditional_thresholds": {
            "any": conditional_thresholds[0],
            "substantial": conditional_thresholds[1],
            "mostly": conditional_thresholds[2],
        },
        "covered_internal": {
            "any_threshold": args.covered_internal_any_threshold,
            "internal_blend_weight": args.internal_blend_weight,
            "loss_weight": args.covered_internal_loss_weight,
            "class_weight_mode": args.covered_internal_class_weight_mode,
            "max_class_weight": args.covered_internal_max_class_weight,
        },
        "conditional_threshold_search_ranges": {
            "any": conditional_any_range,
            "substantial": conditional_substantial_range,
            "mostly": conditional_mostly_range,
        },
        "covered_internal_threshold_search_ranges": {
            "any": covered_internal_any_range,
        },
        "graph_context": {
            "enabled": args.include_graph_features,
            "mode": "structured_features_not_prompt",
            "graph_dir": str(args.graph_dir),
            "canonical_ideas": str(graph_canonical_ideas),
            "graph_file": str(graph_file),
            "relations": sorted(graph_relations),
            "include_direct_target_prior_graph_relations": args.include_direct_target_prior_graph_relations,
            "feature_dim": len(graph_context.feature_names) if graph_context else 0,
            "feature_max_one_hop": args.graph_feature_max_one_hop,
            "feature_max_two_hop_per_neighbor": args.graph_feature_max_two_hop_per_neighbor,
            "feature_gate_init": args.graph_feature_gate_init,
            "loaded": bool(graph_context.loaded) if graph_context else False,
            "edge_count": int(graph_context.edge_count) if graph_context else 0,
            "relation_counts": dict(graph_context.relation_counts) if graph_context else {},
            "feature_names": graph_context.feature_names if graph_context else [],
        },
        "threshold_search_constraints": {
            "min_not_covered_recall": args.threshold_min_not_recall,
            "min_weak_recall": args.threshold_min_weak_recall,
            "min_partial_recall": args.threshold_min_partial_recall,
            "min_mostly_recall": args.threshold_min_mostly_recall,
            "min_covered_recall": args.threshold_min_covered_recall,
            "max_partial_pred_ratio": args.threshold_max_partial_pred_ratio,
            "max_mostly_pred_ratio": args.threshold_max_mostly_pred_ratio,
            "max_partial_pred_multiplier": args.threshold_max_partial_pred_multiplier,
            "max_mostly_pred_multiplier": args.threshold_max_mostly_pred_multiplier,
        },
        "split_source": split_source,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "train_class_distribution": dict(Counter(JOINT_LABELS[int(row["label"])] for row in datasets["train"].items)),
        "expected_sampled_class_distribution": expected_sampler_distribution(datasets["train"], args.balanced_sampling),
        "train_class_score_means": {
            JOINT_LABELS[idx]: round(float(value), 6)
            for idx, value in enumerate(class_score_means)
        },
        "class_weights": [round(float(x), 4) for x in ce_weights.detach().cpu().tolist()],
        "ordinal_pos_weights": [round(float(x), 4) for x in ordinal_weights.detach().cpu().tolist()],
        "covered_internal_class_weights": [
            round(float(x), 4)
            for x in covered_internal_weights.detach().cpu().tolist()
        ],
        "sampler": {
            "balanced_sampling": args.balanced_sampling,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer_steps_per_epoch": math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps),
        },
        "missing_prior_texts": {split: dataset.missing_prior_texts for split, dataset in datasets.items()},
        "missing_pair_targets": {split: dataset.missing_pair_targets for split, dataset in datasets.items()},
        "missing_graph_features": {split: dataset.missing_graph_context for split, dataset in datasets.items()},
        "leakage_control": [
            "gold joint labels/scores are used only as bag-level supervision",
            "gold pair signals are excluded from prompts unless --include-gold-pair-signals is set",
            "pair auxiliary targets are loss-only and can be disabled with --pair-aux-target none",
            "graph structure is encoded as numeric features and is never appended to text prompts",
            "direct target-prior graph relations are excluded from graph features unless --include-direct-target-prior-graph-relations is set",
        ],
    }
    write_json_file(args.output_dir / "run_config.json", run_config)

    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    global_step = 0
    checkpoint_path: Path
    if args.eval_only_checkpoint is None:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running: dict[str, list[float]] = defaultdict(list)
            sampled_labels: list[int] = []
            progress = tqdm(loaders["train"], desc=f"train epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress, start=1):
                sampled_labels.extend(int(label) for label in batch["label"].tolist())
                inputs, pair_mask, graph_features = move_inputs_to_device(batch, device)
                with amp_context(device, args.amp, args.bf16):
                    outputs = model(inputs, pair_mask, graph_features)
                    loss, parts = compute_loss(outputs, batch, ce_weights, ordinal_weights, covered_internal_weights, args)
                if not torch.isfinite(loss).all():
                    optimizer.zero_grad(set_to_none=True)
                    print(json.dumps({"warning": "non_finite_loss", "epoch": epoch, "step": step, "loss_parts": parts}, ensure_ascii=False))
                    continue
                (loss / args.gradient_accumulation_steps).backward()
                should_step = step % args.gradient_accumulation_steps == 0 or step == len(loaders["train"])
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                for name, value in parts.items():
                    running[name].append(value)
                if step % args.log_every == 0:
                    progress.set_postfix(
                        loss=round(float(np.mean(running["total"][-args.log_every:])), 4),
                        hier=round(float(np.mean(running["hierarchical"][-args.log_every:])), 4),
                        score=round(float(np.mean(running["score"][-args.log_every:])), 4),
                        pair=round(float(np.mean(running["pair_aux"][-args.log_every:])), 4),
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

            dev_metrics = evaluate(
                model,
                loaders["dev"],
                device,
                ce_weights,
                ordinal_weights,
                covered_internal_weights,
                thresholds,
                conditional_thresholds,
                args,
            )
            dev_metrics["epoch"] = epoch
            dev_metrics["global_step"] = global_step
            dev_metrics["train_loss"] = {name: round(float(np.mean(values)), 6) for name, values in running.items()}
            sampled_counts = Counter(sampled_labels)
            dev_metrics["sampled_train_distribution"] = {
                JOINT_LABELS[idx]: int(sampled_counts.get(idx, 0))
                for idx in range(len(JOINT_LABELS))
            }
            sampled_partial = int(sampled_counts.get(2, 0))
            sampled_mostly = int(sampled_counts.get(3, 0))
            dev_metrics["sampled_train_conditional_distribution"] = {
                "covered_count": int(sum(sampled_counts.get(idx, 0) for idx in [1, 2, 3])),
                "substantial_count": int(sampled_partial + sampled_mostly),
                "partial_count": sampled_partial,
                "mostly_count": sampled_mostly,
                "partial_to_mostly_ratio": round(float(sampled_partial / max(1, sampled_mostly)), 4),
            }
            print(json.dumps({"split": "dev", **dev_metrics}, ensure_ascii=False, indent=2))
            write_json_file(args.output_dir / f"dev_epoch_{epoch}.json", dev_metrics)

            score = metric_value(dev_metrics, args.selection_metric)
            if score > best_score:
                best_score = score
                best_metrics = dev_metrics
                best_dir = args.output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": run_config,
                        "labels": JOINT_LABELS,
                        "metrics": dev_metrics,
                    },
                    best_dir / "checkpoint.pt",
                )
                tokenizer.save_pretrained(best_dir / "tokenizer")
                write_json_file(best_dir / "metrics.json", dev_metrics)
        checkpoint_path = args.output_dir / "best" / "checkpoint.pt"
    else:
        checkpoint_path = args.eval_only_checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Eval-only checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_metrics = checkpoint.get("metrics")
        if isinstance(checkpoint_metrics, dict):
            best_metrics = checkpoint_metrics
            best_score = metric_value(best_metrics, args.selection_metric)
        else:
            best_metrics = evaluate(
                model,
                loaders["dev"],
                device,
                ce_weights,
                ordinal_weights,
                covered_internal_weights,
                thresholds,
                conditional_thresholds,
                args,
            )
            best_score = metric_value(best_metrics, args.selection_metric)
        best_dir = args.output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        exported_checkpoint = best_dir / "checkpoint.pt"
        shutil.copy2(checkpoint_path, exported_checkpoint)
        tokenizer.save_pretrained(best_dir / "tokenizer")
        write_json_file(best_dir / "metrics.json", best_metrics)
        print(
            json.dumps(
                {
                    "mode": "eval_only",
                    "source_checkpoint": str(checkpoint_path),
                    "exported_checkpoint": str(exported_checkpoint),
                    "selection_metric": args.selection_metric,
                    "selection_score": round(float(best_score), 6),
                },
                ensure_ascii=False,
            )
        )
        checkpoint_path = exported_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_thresholds = thresholds
    final_conditional_thresholds = conditional_thresholds
    final_hier_class_blend_weight = float(args.hier_class_blend_weight)
    final_covered_internal_any_threshold = float(args.covered_internal_any_threshold)
    final_internal_blend_weight = float(args.internal_blend_weight)
    threshold_search: dict[str, Any] | None = None
    conditional_threshold_search: dict[str, Any] | None = None
    covered_internal_threshold_search: dict[str, Any] | None = None
    internal_blend_search: dict[str, Any] | None = None
    hier_class_blend_search: dict[str, Any] | None = None
    if (
        args.tune_thresholds_on_dev
        or args.tune_conditional_thresholds_on_dev
        or args.tune_covered_internal_threshold_on_dev
        or args.tune_internal_blend_on_dev
        or args.tune_hier_class_blend_on_dev
    ):
        dev_probe = evaluate(
            model,
            loaders["dev"],
            device,
            ce_weights,
            ordinal_weights,
            covered_internal_weights,
            thresholds,
            conditional_thresholds,
            args,
            return_predictions=True,
        )
        dev_rows = dev_probe.pop("_predictions")
        dev_true = np.asarray(
            [JOINT_TO_ID.get(str(row.get("gold_joint_coverage_label") or "not_covered"), 0) for row in dev_rows],
            dtype=np.int64,
        )
        dev_scores = np.asarray([safe_float(row.get("pred_joint_coverage_score"), 0.0) for row in dev_rows], dtype=np.float32)
        dev_any_prob = np.asarray(
            [safe_float(row.get("pred_any_coverage_probability"), 0.0) for row in dev_rows],
            dtype=np.float32,
        )
    if args.tune_thresholds_on_dev:
        final_thresholds, threshold_metrics = tune_ordered_thresholds(
            dev_true,
            dev_scores,
            args.threshold_search_metric,
            args.threshold_steps,
            thresholds,
            args.threshold_min_not_recall,
            args.threshold_min_weak_recall,
            args.threshold_min_partial_recall,
            args.threshold_min_mostly_recall,
            args.threshold_min_covered_recall,
            args.threshold_max_partial_pred_ratio,
            args.threshold_max_mostly_pred_ratio,
            args.threshold_max_partial_pred_multiplier,
            args.threshold_max_mostly_pred_multiplier,
        )
        threshold_search = {
            "source_split": "dev",
            "initial_thresholds": {
                "weak": thresholds[0],
                "partial": thresholds[1],
                "mostly": thresholds[2],
            },
            "tuned_thresholds": {
                "weak": round(float(final_thresholds[0]), 6),
                "partial": round(float(final_thresholds[1]), 6),
                "mostly": round(float(final_thresholds[2]), 6),
            },
            "dev_metrics": threshold_metrics,
        }
        write_json_file(args.output_dir / "dev_threshold_search.json", threshold_search)
        print(json.dumps({"split": "dev_threshold_search", **threshold_search}, ensure_ascii=False, indent=2))
    if args.tune_conditional_thresholds_on_dev:
        dev_substantial_prob = np.asarray(
            [safe_float(row.get("pred_substantial_probability_given_covered"), 0.0) for row in dev_rows],
            dtype=np.float32,
        )
        dev_mostly_prob = np.asarray(
            [safe_float(row.get("pred_mostly_probability_given_substantial"), 0.0) for row in dev_rows],
            dtype=np.float32,
        )
        final_conditional_thresholds, conditional_threshold_metrics = tune_conditional_thresholds(
            dev_true,
            dev_any_prob,
            dev_substantial_prob,
            dev_mostly_prob,
            args.threshold_search_metric,
            args.threshold_steps,
            conditional_thresholds,
            conditional_any_range,
            conditional_substantial_range,
            conditional_mostly_range,
            args.threshold_min_not_recall,
            args.threshold_min_weak_recall,
            args.threshold_min_partial_recall,
            args.threshold_min_mostly_recall,
            args.threshold_min_covered_recall,
            args.threshold_max_partial_pred_ratio,
            args.threshold_max_mostly_pred_ratio,
            args.threshold_max_partial_pred_multiplier,
            args.threshold_max_mostly_pred_multiplier,
        )
        conditional_threshold_search = {
            "source_split": "dev",
            "initial_thresholds": {
                "any": conditional_thresholds[0],
                "substantial": conditional_thresholds[1],
                "mostly": conditional_thresholds[2],
            },
            "tuned_thresholds": {
                "any": round(float(final_conditional_thresholds[0]), 6),
                "substantial": round(float(final_conditional_thresholds[1]), 6),
                "mostly": round(float(final_conditional_thresholds[2]), 6),
            },
            "search_ranges": {
                "any": conditional_any_range,
                "substantial": conditional_substantial_range,
                "mostly": conditional_mostly_range,
            },
            "dev_metrics": conditional_threshold_metrics,
        }
        write_json_file(args.output_dir / "dev_conditional_threshold_search.json", conditional_threshold_search)
        print(json.dumps({"split": "dev_conditional_threshold_search", **conditional_threshold_search}, ensure_ascii=False, indent=2))
    if args.tune_covered_internal_threshold_on_dev:
        dev_covered_internal_probs = covered_internal_probability_matrix(
            dev_rows,
            "pred_covered_internal_conditional_probabilities",
        )
        final_covered_internal_any_threshold, covered_internal_threshold_metrics = tune_covered_internal_threshold(
            dev_true,
            dev_any_prob,
            dev_covered_internal_probs,
            args.threshold_search_metric,
            args.threshold_steps,
            args.covered_internal_any_threshold,
            covered_internal_any_range,
            args.threshold_min_not_recall,
            args.threshold_min_weak_recall,
            args.threshold_min_partial_recall,
            args.threshold_min_mostly_recall,
            args.threshold_min_covered_recall,
            args.threshold_max_partial_pred_ratio,
            args.threshold_max_mostly_pred_ratio,
            args.threshold_max_partial_pred_multiplier,
            args.threshold_max_mostly_pred_multiplier,
        )
        args.covered_internal_any_threshold = final_covered_internal_any_threshold
        covered_internal_threshold_search = {
            "source_split": "dev",
            "initial_threshold": round(float(run_config["args"]["covered_internal_any_threshold"]), 6),
            "tuned_threshold": round(float(final_covered_internal_any_threshold), 6),
            "search_range": covered_internal_any_range,
            "dev_metrics": covered_internal_threshold_metrics,
        }
        write_json_file(args.output_dir / "dev_covered_internal_threshold_search.json", covered_internal_threshold_search)
        print(json.dumps({"split": "dev_covered_internal_threshold_search", **covered_internal_threshold_search}, ensure_ascii=False, indent=2))
    if args.tune_internal_blend_on_dev:
        dev_hier_probs = prediction_probability_matrix(dev_rows, "pred_hierarchical_joint_probabilities")
        dev_internal_probs = prediction_probability_matrix(dev_rows, "pred_covered_internal_joint_probabilities")
        final_internal_blend_weight, internal_blend_metrics = tune_hier_class_blend(
            dev_true,
            dev_hier_probs,
            dev_internal_probs,
            args.threshold_search_metric,
            args.threshold_steps,
            args.internal_blend_weight,
            args.threshold_min_not_recall,
            args.threshold_min_weak_recall,
            args.threshold_min_partial_recall,
            args.threshold_min_mostly_recall,
            args.threshold_min_covered_recall,
            args.threshold_max_partial_pred_ratio,
            args.threshold_max_mostly_pred_ratio,
            args.threshold_max_partial_pred_multiplier,
            args.threshold_max_mostly_pred_multiplier,
        )
        args.internal_blend_weight = final_internal_blend_weight
        internal_blend_search = {
            "source_split": "dev",
            "initial_internal_weight": round(float(run_config["args"]["internal_blend_weight"]), 6),
            "tuned_internal_weight": round(float(final_internal_blend_weight), 6),
            "dev_metrics": internal_blend_metrics,
        }
        write_json_file(args.output_dir / "dev_internal_blend_search.json", internal_blend_search)
        print(json.dumps({"split": "dev_internal_blend_search", **internal_blend_search}, ensure_ascii=False, indent=2))
    if args.tune_hier_class_blend_on_dev:
        dev_hier_probs = prediction_probability_matrix(dev_rows, "pred_hierarchical_joint_probabilities")
        dev_class_probs = prediction_probability_matrix(dev_rows, "pred_class_head_joint_probabilities")
        final_hier_class_blend_weight, blend_metrics = tune_hier_class_blend(
            dev_true,
            dev_hier_probs,
            dev_class_probs,
            args.threshold_search_metric,
            args.threshold_steps,
            args.hier_class_blend_weight,
            args.threshold_min_not_recall,
            args.threshold_min_weak_recall,
            args.threshold_min_partial_recall,
            args.threshold_min_mostly_recall,
            args.threshold_min_covered_recall,
            args.threshold_max_partial_pred_ratio,
            args.threshold_max_mostly_pred_ratio,
            args.threshold_max_partial_pred_multiplier,
            args.threshold_max_mostly_pred_multiplier,
        )
        args.hier_class_blend_weight = final_hier_class_blend_weight
        hier_class_blend_search = {
            "source_split": "dev",
            "initial_class_weight": round(float(run_config["args"]["hier_class_blend_weight"]), 6),
            "tuned_class_weight": round(float(final_hier_class_blend_weight), 6),
            "dev_metrics": blend_metrics,
        }
        write_json_file(args.output_dir / "dev_hier_class_blend_search.json", hier_class_blend_search)
        print(json.dumps({"split": "dev_hier_class_blend_search", **hier_class_blend_search}, ensure_ascii=False, indent=2))

    final_metrics: dict[str, Any] = {
        "best_dev": best_metrics,
        "threshold_search": threshold_search,
        "conditional_threshold_search": conditional_threshold_search,
        "covered_internal_threshold_search": covered_internal_threshold_search,
        "internal_blend_search": internal_blend_search,
        "hier_class_blend_search": hier_class_blend_search,
        "final_label_thresholds": {
            "weak": round(float(final_thresholds[0]), 6),
            "partial": round(float(final_thresholds[1]), 6),
            "mostly": round(float(final_thresholds[2]), 6),
        },
        "final_conditional_thresholds": {
            "any": round(float(final_conditional_thresholds[0]), 6),
            "substantial": round(float(final_conditional_thresholds[1]), 6),
            "mostly": round(float(final_conditional_thresholds[2]), 6),
        },
        "final_hier_class_blend_weight": round(float(final_hier_class_blend_weight), 6),
        "final_covered_internal_any_threshold": round(float(final_covered_internal_any_threshold), 6),
        "final_internal_blend_weight": round(float(final_internal_blend_weight), 6),
    }
    for split in ["train", "dev", "test"]:
        metrics = evaluate(
            model,
            final_loaders[split],
            device,
            ce_weights,
            ordinal_weights,
            covered_internal_weights,
            final_thresholds,
            final_conditional_thresholds,
            args,
            return_predictions=True,
        )
        pred_rows = metrics.pop("_predictions")
        expected_ids = Counter(str(item["target_idea_id"]) for item in datasets[split].items)
        predicted_ids = Counter(str(row.get("target_idea_id") or "") for row in pred_rows)
        if predicted_ids != expected_ids:
            missing = list((expected_ids - predicted_ids).elements())[:10]
            extra = list((predicted_ids - expected_ids).elements())[:10]
            raise RuntimeError(
                f"Non-exhaustive {split} prediction export: "
                f"expected_rows={sum(expected_ids.values())}, "
                f"predicted_rows={sum(predicted_ids.values())}, "
                f"missing_examples={missing}, extra_examples={extra}"
            )
        final_metrics[split] = metrics
        write_jsonl(args.output_dir / f"{split}_predictions.jsonl", pred_rows)
        print(json.dumps({"split": split, **metrics}, ensure_ascii=False, indent=2))
    write_json_file(args.output_dir / "metrics.json", final_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
