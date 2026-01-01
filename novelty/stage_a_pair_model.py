from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


COVERAGE_LEVELS = ["not_covering", "partial_cover", "large_cover", "full_cover"]
BASE_PAIR_LABELS = ["not_covering", "related_not_covering", "partial_cover", "large_cover"]
FULL_PAIR_LABELS = BASE_PAIR_LABELS + ["full_cover"]
DEFAULT_PAIR_SCORE_BY_LABEL = {
    "not_covering": 0.1,
    "related_not_covering": 0.34,
    "partial_cover": 0.6,
    "large_cover": 0.85,
    "full_cover": 1.0,
}
EVIDENCE_PAIR_SCORE_BY_LABEL = {
    "not_covering": 0.0,
    "related_not_covering": 0.0,
    "partial_cover": 0.5,
    "large_cover": 1.0,
    "full_cover": 1.0,
}

COVERAGE_TO_LEVEL = {
    "not_covering": 0,
    "related_not_covering": 0,
    "partial_cover": 1,
    "large_cover": 2,
    "full_cover": 3,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def temporal_split(rows: list[dict[str, Any]], date_key: str = "target_date") -> tuple[list[int], list[int], list[int]]:
    dated = [(idx, int(row.get(date_key) or 0)) for idx, row in enumerate(rows)]
    dated = [(idx, date) for idx, date in dated if date > 0]
    dated.sort(key=lambda x: (x[1], x[0]))
    n = len(dated)
    return (
        [idx for idx, _ in dated[: int(n * 0.70)]],
        [idx for idx, _ in dated[int(n * 0.70): int(n * 0.85)]],
        [idx for idx, _ in dated[int(n * 0.85):]],
    )


def months_between(later: int, earlier: int) -> int:
    ly, lm = divmod(int(later), 100)
    ey, em = divmod(int(earlier), 100)
    return max(0, (ly - ey) * 12 + (lm - em))


def coverage_level(row: dict[str, Any]) -> int:
    return COVERAGE_TO_LEVEL.get(str(row.get("coverage_label") or "not_covering"), 0)


def relatedness(row: dict[str, Any]) -> int:
    label = row.get("relatedness_label")
    if label in {"related", "unrelated"}:
        return 1 if label == "related" else 0
    return 0 if row.get("coverage_label") == "not_covering" else 1


def pair_label_names(rows: list[dict[str, Any]]) -> list[str]:
    has_full = any(row.get("coverage_label") == "full_cover" for row in rows)
    return FULL_PAIR_LABELS if has_full else BASE_PAIR_LABELS


def pair_label_id(row: dict[str, Any], labels: list[str]) -> int:
    label = str(row.get("coverage_label") or "not_covering")
    if label == "full_cover" and label not in labels:
        label = "large_cover"
    try:
        return labels.index(label)
    except ValueError:
        return 0


def ordinal_targets(labels: torch.Tensor, num_thresholds: int = 3) -> torch.Tensor:
    thresholds = torch.arange(num_thresholds, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    observed = np.zeros((n_classes, n_classes), dtype=np.float64)
    for true, pred in zip(y_true, y_pred, strict=True):
        observed[int(true), int(pred)] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(1.0, observed.sum())
    weights = np.zeros_like(observed)
    denom = max(1, (n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / denom
    den = float((weights * expected).sum())
    return float(1 - (weights * observed).sum() / den) if den else 0.0


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    out = {}
    for idx, label in enumerate(labels):
        true_pos = int(((y_true == idx) & (y_pred == idx)).sum())
        false_pos = int(((y_true != idx) & (y_pred == idx)).sum())
        false_neg = int(((y_true == idx) & (y_pred != idx)).sum())
        precision = true_pos / max(1, true_pos + false_pos)
        recall = true_pos / max(1, true_pos + false_neg)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int((y_true == idx).sum()),
        }
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], proba: np.ndarray | None = None) -> dict[str, Any]:
    per_class = per_class_metrics(y_true, y_pred, labels)
    f1s = [item["f1"] for item in per_class.values()]
    out = {
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "ordinal_mae": round(float(np.abs(y_true - y_pred).mean()), 4) if len(y_true) else 0.0,
        "qwk": round(qwk(y_true, y_pred, len(labels)), 4) if len(y_true) else 0.0,
        "per_class": per_class,
    }
    if proba is not None and len(proba):
        out["mean_confidence"] = round(float(proba.max(axis=1).mean()), 4)
    covering_ids = {
        idx
        for idx, label in enumerate(labels)
        if label in {"partial_cover", "large_cover", "full_cover"}
    }
    if covering_ids:
        true_covering = np.asarray([int(value) in covering_ids for value in y_true], dtype=bool)
        pred_covering = np.asarray([int(value) in covering_ids for value in y_pred], dtype=bool)
        tp = int((true_covering & pred_covering).sum())
        fp = int((~true_covering & pred_covering).sum())
        fn = int((true_covering & ~pred_covering).sum())
        tn = int((~true_covering & ~pred_covering).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out["covering_binary"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "false_positive_rate": round(float(fp / max(1, fp + tn)), 4),
            "false_negative_rate": round(float(fn / max(1, fn + tp)), 4),
        }
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if not len(y_true):
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0}
    err = y_pred - y_true
    centered_true = y_true - y_true.mean()
    centered_pred = y_pred - y_pred.mean()
    denom = float(np.sqrt((centered_true**2).sum()) * np.sqrt((centered_pred**2).sum()))
    pearson = float((centered_true * centered_pred).sum() / denom) if denom else 0.0
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err**2).mean())), 4),
        "pearson": round(pearson, 4),
    }


def pairwise_ranking_metrics(rows: list[dict[str, Any]], score_pred: np.ndarray, indices: list[int], margin: float) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for local_idx, row_idx in enumerate(indices):
        row = rows[row_idx]
        grouped[str(row.get("target_idea_id") or "")].append(
            (local_idx, float(row.get("coverage_score") or 0.0), float(score_pred[local_idx]))
        )
    correct = 0
    total = 0
    for items in grouped.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                _, gold_i, pred_i = items[i]
                _, gold_j, pred_j = items[j]
                if abs(gold_i - gold_j) < margin:
                    continue
                total += 1
                if (gold_i > gold_j and pred_i > pred_j) or (gold_j > gold_i and pred_j > pred_i):
                    correct += 1
    return {
        "target_groups": len(grouped),
        "pairwise_comparisons": total,
        "pairwise_accuracy": round(correct / max(1, total), 4),
        "margin": margin,
    }


def pair_score_levels_from_rows(
    rows: list[dict[str, Any]],
    indices: list[int],
    pair_labels: list[str],
    score_mode: str,
) -> list[float]:
    if score_mode == "coverage_evidence":
        return [
            EVIDENCE_PAIR_SCORE_BY_LABEL["large_cover" if label == "full_cover" else label]
            for label in pair_labels
        ]

    values: dict[str, list[float]] = defaultdict(list)
    for idx in indices:
        label = str(rows[idx].get("coverage_label") or "not_covering")
        if label == "full_cover" and label not in pair_labels:
            label = "large_cover"
        if label in pair_labels:
            values[label].append(safe_float(rows[idx].get("coverage_score"), DEFAULT_PAIR_SCORE_BY_LABEL.get(label, 0.0)))
    levels = []
    for label in pair_labels:
        label_values = values.get(label) or []
        default = DEFAULT_PAIR_SCORE_BY_LABEL.get(label, 0.0)
        levels.append(float(np.mean(label_values)) if label_values else default)
    return levels


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class PairCoverageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        indices: list[int],
        pair_labels: list[str],
        label_sample_weights: dict[str, float] | None = None,
    ) -> None:
        self.rows = rows
        self.indices = indices
        self.pair_labels = pair_labels
        self.label_sample_weights = label_sample_weights or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = self.indices[item]
        row = self.rows[idx]
        label = str(row.get("coverage_label") or "not_covering")
        sample_weight = min(1.0, max(0.2, safe_float(row.get("label_confidence"), 1.0)))
        sample_weight *= float(self.label_sample_weights.get(label, 1.0))
        return {
            "row_idx": idx,
            "row": row,
            "coverage_level": coverage_level(row),
            "relatedness": relatedness(row),
            "pair_label": pair_label_id(row, self.pair_labels),
            "coverage_score": safe_float(row.get("coverage_score"), 0.0),
            "sample_weight": sample_weight,
        }


class PairCoverageCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def prompt(row: dict[str, Any]) -> str:
        target_date = int(row.get("target_date") or 0)
        prior_date = int(row.get("prior_date") or 0)
        gap = months_between(target_date, prior_date) if target_date and prior_date else 0
        target_meta = [
            f"aspect={row.get('target_primary_aspect') or 'unknown'}",
            f"contribution={row.get('target_contribution_type') or 'unknown'}",
            f"task={row.get('target_task_family') or 'unknown'}",
            f"date={target_date or 'unknown'}",
        ]
        prior_meta = [
            f"aspect={row.get('prior_primary_aspect') or 'unknown'}",
            f"contribution={row.get('prior_contribution_type') or 'unknown'}",
            f"task={row.get('prior_task_family') or 'unknown'}",
            f"date={prior_date or 'unknown'}",
            f"time_gap_months={gap}",
        ]
        return (
            "Judge whether a historical prior substantively covers a target scientific idea.\n"
            "Use semantic content, task/method alignment, and publication time. Do not assume that related work covers the target.\n"
            f"[TARGET METADATA] {'; '.join(target_meta)}\n"
            f"[TARGET IDEA]\n{row.get('target_text') or ''}\n"
            f"[PRIOR METADATA] {'; '.join(prior_meta)}\n"
            f"[HISTORICAL PRIOR]\n{row.get('prior_text') or ''}"
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [self.prompt(item["row"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = {
            "coverage_level": torch.tensor([item["coverage_level"] for item in batch], dtype=torch.long),
            "relatedness": torch.tensor([item["relatedness"] for item in batch], dtype=torch.float32),
            "pair_label": torch.tensor([item["pair_label"] for item in batch], dtype=torch.long),
            "coverage_score": torch.tensor([item["coverage_score"] for item in batch], dtype=torch.float32),
            "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        }
        return {
            "inputs": encoded,
            "labels": labels,
            "row_idx": [item["row_idx"] for item in batch],
        }


@dataclass
class PairCoverageConfig:
    model_name: str
    dropout: float = 0.1
    trust_remote_code: bool = True
    cache_dir: str | None = None
    local_files_only: bool = False


class PairCoverageModel(nn.Module):
    def __init__(self, cfg: PairCoverageConfig) -> None:
        super().__init__()
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
        self.coverage_head = nn.Linear(hidden_size, 3)
        self.relatedness_head = nn.Linear(hidden_size, 1)
        self.covering_head = nn.Linear(hidden_size, 1)
        self.strength_head = nn.Linear(hidden_size, 1)
        self.direct_pair_head = nn.Linear(hidden_size, len(BASE_PAIR_LABELS))
        self.score_head = nn.Linear(hidden_size, 1)

    def pooled(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.dropout(self.pooled(inputs))
        return {
            "coverage_logits": self.coverage_head(h),
            "relatedness_logits": self.relatedness_head(h).squeeze(-1),
            "covering_logits": self.covering_head(h).squeeze(-1),
            "strength_logits": self.strength_head(h).squeeze(-1),
            "direct_pair_logits": self.direct_pair_head(h),
            "score": torch.sigmoid(self.score_head(h).squeeze(-1)),
        }


def move_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def coverage_class_probs(coverage_logits: torch.Tensor) -> torch.Tensor:
    gt = torch.sigmoid(coverage_logits.float())
    gt = torch.cummin(gt, dim=1).values
    p0 = 1.0 - gt[:, 0]
    p1 = (gt[:, 0] - gt[:, 1]).clamp_min(0.0)
    p2 = (gt[:, 1] - gt[:, 2]).clamp_min(0.0)
    p3 = gt[:, 2]
    probs = torch.stack([p0, p1, p2, p3], dim=-1)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def pair_probs(
    related_prob: torch.Tensor,
    coverage_probs: torch.Tensor,
    pair_labels: list[str],
) -> torch.Tensor:
    unrelated = 1.0 - related_prob
    related = related_prob
    if "full_cover" in pair_labels:
        probs = torch.stack(
            [
                unrelated,
                related * coverage_probs[:, 0],
                related * coverage_probs[:, 1],
                related * coverage_probs[:, 2],
                related * coverage_probs[:, 3],
            ],
            dim=-1,
        )
    else:
        probs = torch.stack(
            [
                unrelated,
                related * coverage_probs[:, 0],
                related * coverage_probs[:, 1],
                related * (coverage_probs[:, 2] + coverage_probs[:, 3]),
            ],
            dim=-1,
        )
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def staged_pair_probs(
    related_prob: torch.Tensor,
    covering_prob: torch.Tensor,
    strength_prob: torch.Tensor,
    direct_logits: torch.Tensor,
    direct_blend_weight: float,
) -> torch.Tensor:
    hierarchical = torch.stack(
        [
            1.0 - related_prob,
            related_prob * (1.0 - covering_prob),
            related_prob * covering_prob * (1.0 - strength_prob),
            related_prob * covering_prob * strength_prob,
        ],
        dim=-1,
    )
    hierarchical = hierarchical / hierarchical.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if direct_blend_weight <= 0:
        return hierarchical
    direct = torch.softmax(direct_logits.float(), dim=-1)
    weight = min(1.0, max(0.0, float(direct_blend_weight)))
    probs = (1.0 - weight) * hierarchical + weight * direct
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def coverage_probs_from_pair_probs(pair_prob: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(pair_prob[:, 0])
    probs = torch.stack(
        [
            pair_prob[:, 0] + pair_prob[:, 1],
            pair_prob[:, 2],
            pair_prob[:, 3],
            zeros,
        ],
        dim=-1,
    )
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def scaled_pos_weight(raw: float, scale: float) -> float:
    return max(0.1, 1.0 + float(scale) * (float(raw) - 1.0))


def weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    weights = sample_weight.to(dtype=loss.dtype, device=loss.device)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def pair_evidence_score_target(pair: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(pair, dtype=torch.float32)
    target = target.masked_fill(pair >= 2, 0.5)
    target = target.masked_fill(pair >= 3, 1.0)
    return target


def staged_expected_evidence_score(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    related = torch.sigmoid(outputs["relatedness_logits"].float())
    covering = torch.sigmoid(outputs["covering_logits"].float())
    strength = torch.sigmoid(outputs["strength_logits"].float())
    return 0.5 * related * covering * (1.0 + strength)


def compute_pos_weights(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    device: torch.device,
    cap: float = 10.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    coverage_targets = np.asarray([coverage_level(rows[idx]) for idx in train_idx], dtype=np.int64)
    ordinal = np.stack([(coverage_targets > threshold).astype(np.float32) for threshold in range(3)], axis=1)
    pos = ordinal.sum(axis=0)
    neg = ordinal.shape[0] - pos
    cov_weights = np.ones(3, dtype=np.float32)
    for idx in range(3):
        if pos[idx] > 0:
            raw = min(float(cap), max(0.1, float(neg[idx] / pos[idx])))
            cov_weights[idx] = scaled_pos_weight(raw, scale)

    rel_targets = np.asarray([relatedness(rows[idx]) for idx in train_idx], dtype=np.float32)
    rel_pos = float(rel_targets.sum())
    rel_neg = float(len(rel_targets) - rel_pos)
    rel_weight = scaled_pos_weight(min(float(cap), max(0.1, rel_neg / rel_pos)), scale) if rel_pos else 1.0
    return (
        torch.tensor(cov_weights, dtype=torch.float32, device=device),
        torch.tensor(rel_weight, dtype=torch.float32, device=device),
    )


def _binary_pos_weight(targets: np.ndarray, cap: float = 10.0, scale: float = 1.0) -> float:
    pos = float(targets.sum())
    neg = float(len(targets) - pos)
    return scaled_pos_weight(min(cap, max(0.1, neg / pos)), scale) if pos else 1.0


def compute_staged_weights(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    pair_labels: list[str],
    device: torch.device,
    cap: float = 10.0,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    pair_targets = np.asarray([pair_label_id(rows[idx], pair_labels) for idx in train_idx], dtype=np.int64)
    related_targets = (pair_targets > 0).astype(np.float32)

    related_idx = pair_targets > 0
    covering_targets = (pair_targets[related_idx] >= 2).astype(np.float32)

    covering_idx = pair_targets >= 2
    strength_targets = (pair_targets[covering_idx] == 3).astype(np.float32)

    counts = Counter(int(x) for x in pair_targets)
    total = max(1, len(pair_targets))
    direct_weights = []
    for class_id in range(len(BASE_PAIR_LABELS)):
        value = total / max(1, len(BASE_PAIR_LABELS) * counts.get(class_id, 0))
        direct_weights.append(scaled_pos_weight(min(5.0, max(0.1, value)), scale))

    return {
        "related": torch.tensor(_binary_pos_weight(related_targets, cap, scale), dtype=torch.float32, device=device),
        "covering": torch.tensor(_binary_pos_weight(covering_targets, cap, scale), dtype=torch.float32, device=device),
        "strength": torch.tensor(_binary_pos_weight(strength_targets, cap, scale), dtype=torch.float32, device=device),
        "direct": torch.tensor(direct_weights, dtype=torch.float32, device=device),
    }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    coverage_pos_weight: torch.Tensor,
    related_pos_weight: torch.Tensor,
    coverage_weight: float,
    related_weight: float,
    score_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_weight = labels["sample_weight"]
    cov_targets = ordinal_targets(labels["coverage_level"])
    cov_loss = F.binary_cross_entropy_with_logits(
        outputs["coverage_logits"],
        cov_targets,
        pos_weight=coverage_pos_weight,
        reduction="none",
    ).mean(dim=-1)
    cov_loss = weighted_mean(cov_loss, sample_weight)
    related_loss = F.binary_cross_entropy_with_logits(
        outputs["relatedness_logits"],
        labels["relatedness"],
        pos_weight=related_pos_weight,
        reduction="none",
    )
    related_loss = weighted_mean(related_loss, sample_weight)
    score_loss = F.mse_loss(outputs["score"], labels["coverage_score"], reduction="none")
    score_loss = weighted_mean(score_loss, sample_weight)
    total = coverage_weight * cov_loss + related_weight * related_loss + score_weight * score_loss
    return total, {
        "total": float(total.detach().cpu()),
        "coverage_ordinal": float(cov_loss.detach().cpu()),
        "relatedness_bce": float(related_loss.detach().cpu()),
        "score_mse": float(score_loss.detach().cpu()),
    }


def compute_staged_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    related_weight: float,
    covering_weight: float,
    strength_weight: float,
    direct_weight: float,
    score_weight: float,
    expected_score_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_weight = labels["sample_weight"]
    pair = labels["pair_label"]

    related_loss = F.binary_cross_entropy_with_logits(
        outputs["relatedness_logits"],
        labels["relatedness"],
        pos_weight=weights["related"],
        reduction="none",
    )
    related_loss = weighted_mean(related_loss, sample_weight)

    related_mask = pair > 0
    covering_target = (pair >= 2).float()
    if related_mask.any():
        covering_loss = F.binary_cross_entropy_with_logits(
            outputs["covering_logits"][related_mask],
            covering_target[related_mask],
            pos_weight=weights["covering"],
            reduction="none",
        )
        covering_loss = weighted_mean(covering_loss, sample_weight[related_mask])
    else:
        covering_loss = outputs["covering_logits"].sum() * 0.0

    covering_mask = pair >= 2
    strength_target = (pair == 3).float()
    if covering_mask.any():
        strength_loss = F.binary_cross_entropy_with_logits(
            outputs["strength_logits"][covering_mask],
            strength_target[covering_mask],
            pos_weight=weights["strength"],
            reduction="none",
        )
        strength_loss = weighted_mean(strength_loss, sample_weight[covering_mask])
    else:
        strength_loss = outputs["strength_logits"].sum() * 0.0

    direct_loss = F.cross_entropy(
        outputs["direct_pair_logits"],
        pair.clamp(0, len(BASE_PAIR_LABELS) - 1),
        weight=weights["direct"],
        reduction="none",
    )
    direct_loss = weighted_mean(direct_loss, sample_weight)

    score_loss = F.mse_loss(outputs["score"], labels["coverage_score"], reduction="none")
    score_loss = weighted_mean(score_loss, sample_weight)

    expected_score_target = pair_evidence_score_target(pair)
    expected_score_loss = F.mse_loss(
        staged_expected_evidence_score(outputs),
        expected_score_target,
        reduction="none",
    )
    expected_score_loss = weighted_mean(expected_score_loss, sample_weight)

    total = (
        related_weight * related_loss
        + covering_weight * covering_loss
        + strength_weight * strength_loss
        + direct_weight * direct_loss
        + score_weight * score_loss
        + expected_score_weight * expected_score_loss
    )
    return total, {
        "total": float(total.detach().cpu()),
        "relatedness_bce": float(related_loss.detach().cpu()),
        "covering_bce": float(covering_loss.detach().cpu()),
        "strength_bce": float(strength_loss.detach().cpu()),
        "direct_pair_ce": float(direct_loss.detach().cpu()),
        "score_mse": float(score_loss.detach().cpu()),
        "pair_expected_mse": float(expected_score_loss.detach().cpu()),
    }


def device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool, bf16: bool) -> torch.autocast:
    device_type = "cuda" if device.type == "cuda" else "mps" if device.type == "mps" else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled and device.type != "cpu")


@torch.no_grad()
def predict(
    model: PairCoverageModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    pair_labels: list[str],
    pair_score_levels: list[float],
    label_schema: str,
    direct_blend_weight: float,
    amp: bool,
    bf16: bool,
) -> dict[str, Any]:
    model.eval()
    row_idx: list[int] = []
    related_true: list[int] = []
    coverage_true: list[int] = []
    pair_true: list[int] = []
    score_true: list[float] = []
    related_prob_all: list[float] = []
    coverage_prob_all: list[list[float]] = []
    pair_prob_all: list[list[float]] = []
    score_pred_all: list[float] = []
    covering_prob_all: list[float] = []
    strength_prob_all: list[float] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        labels = {key: value.to(device) for key, value in batch["labels"].items()}
        inputs = move_to_device(batch["inputs"], device)
        with amp_context(device, amp, bf16):
            outputs = model(inputs)
        related_prob = torch.sigmoid(outputs["relatedness_logits"].float())
        if label_schema == "staged":
            covering_prob = torch.sigmoid(outputs["covering_logits"].float())
            strength_prob = torch.sigmoid(outputs["strength_logits"].float())
            pair_prob = staged_pair_probs(
                related_prob,
                covering_prob,
                strength_prob,
                outputs["direct_pair_logits"],
                direct_blend_weight,
            )
            coverage_probs = coverage_probs_from_pair_probs(pair_prob)
            covering_prob_all.extend(float(x) for x in covering_prob.detach().cpu().numpy().tolist())
            strength_prob_all.extend(float(x) for x in strength_prob.detach().cpu().numpy().tolist())
        else:
            coverage_probs = coverage_class_probs(outputs["coverage_logits"])
            pair_prob = pair_probs(related_prob, coverage_probs, pair_labels)
            covering_prob_all.extend(float(x) for x in (coverage_probs[:, 1] + coverage_probs[:, 2]).detach().cpu().numpy().tolist())
            strength_prob_all.extend(float(x) for x in coverage_probs[:, 2].detach().cpu().numpy().tolist())

        row_idx.extend(int(idx) for idx in batch["row_idx"])
        related_true.extend(int(x) for x in labels["relatedness"].detach().cpu().numpy().tolist())
        coverage_true.extend(int(x) for x in labels["coverage_level"].detach().cpu().numpy().tolist())
        pair_true.extend(int(x) for x in labels["pair_label"].detach().cpu().numpy().tolist())
        score_true.extend(float(x) for x in labels["coverage_score"].detach().cpu().numpy().tolist())
        related_prob_all.extend(float(x) for x in related_prob.detach().cpu().numpy().tolist())
        coverage_prob_all.extend(coverage_probs.detach().cpu().numpy().astype(float).tolist())
        pair_prob_all.extend(pair_prob.detach().cpu().numpy().astype(float).tolist())
        score_pred_all.extend(float(x) for x in outputs["score"].detach().float().cpu().numpy().tolist())

    related_prob_np = np.asarray(related_prob_all, dtype=np.float32)
    coverage_prob_np = np.asarray(coverage_prob_all, dtype=np.float32)
    pair_prob_np = np.asarray(pair_prob_all, dtype=np.float32)
    score_pred_np = np.asarray(score_pred_all, dtype=np.float32)
    score_levels_np = np.asarray(pair_score_levels, dtype=np.float32)
    if len(score_levels_np) != pair_prob_np.shape[1]:
        score_levels_np = np.linspace(0.0, 1.0, pair_prob_np.shape[1], dtype=np.float32)
    pair_expected_score_np = pair_prob_np @ score_levels_np
    related_true_np = np.asarray(related_true, dtype=np.int64)
    coverage_true_np = np.asarray(coverage_true, dtype=np.int64)
    pair_true_np = np.asarray(pair_true, dtype=np.int64)
    score_true_np = np.asarray(score_true, dtype=np.float32)
    return {
        "row_idx": row_idx,
        "related_true": related_true_np,
        "coverage_true": coverage_true_np,
        "pair_true": pair_true_np,
        "score_true": score_true_np,
        "related_prob": related_prob_np,
        "coverage_prob": coverage_prob_np,
        "pair_prob": pair_prob_np,
        "score_pred": score_pred_np,
        "pair_expected_score": pair_expected_score_np.astype(np.float32),
        "covering_prob": np.asarray(covering_prob_all, dtype=np.float32),
        "strength_prob": np.asarray(strength_prob_all, dtype=np.float32),
        "related_pred": (related_prob_np >= 0.5).astype(np.int64),
        "coverage_pred": coverage_prob_np.argmax(axis=1).astype(np.int64),
        "pair_pred": pair_prob_np.argmax(axis=1).astype(np.int64),
    }


def score_predictions(pred: dict[str, Any], source: str) -> np.ndarray:
    if source == "pair_expected":
        return pred["pair_expected_score"]
    return pred["score_pred"]


def split_metrics(
    rows: list[dict[str, Any]],
    indices: list[int],
    pred: dict[str, Any],
    pair_labels: list[str],
    ranking_margin: float,
    score_source: str,
) -> dict[str, Any]:
    related_proba = np.stack([1.0 - pred["related_prob"], pred["related_prob"]], axis=1)
    score_pred = score_predictions(pred, score_source)
    metrics = {
        "rows": len(indices),
        "score_source": score_source,
        "relatedness": classification_metrics(
            pred["related_true"],
            pred["related_pred"],
            ["unrelated", "related"],
            related_proba,
        ),
        "coverage_ordinal": classification_metrics(
            pred["coverage_true"],
            pred["coverage_pred"],
            COVERAGE_LEVELS,
            pred["coverage_prob"],
        ),
        "pair_combined": classification_metrics(
            pred["pair_true"],
            pred["pair_pred"],
            pair_labels,
            pred["pair_prob"],
        ),
        "coverage_score": regression_metrics(pred["score_true"], score_pred),
        "score_head_coverage_score": regression_metrics(pred["score_true"], pred["score_pred"]),
        "pair_expected_coverage_score": regression_metrics(pred["score_true"], pred["pair_expected_score"]),
        "ranking": pairwise_ranking_metrics(rows, score_pred, indices, ranking_margin),
        "pred_distribution": dict(Counter(pair_labels[int(idx)] for idx in pred["pair_pred"])),
        "true_distribution": dict(Counter(pair_labels[int(idx)] for idx in pred["pair_true"])),
    }
    return metrics


def prediction_rows(
    rows: list[dict[str, Any]],
    split: str,
    pred: dict[str, Any],
    pair_labels: list[str],
    score_source: str,
) -> list[dict[str, Any]]:
    out = []
    exported_score = score_predictions(pred, score_source)
    for local_idx, row_idx in enumerate(pred["row_idx"]):
        row = rows[row_idx]
        pair_pred = int(pred["pair_pred"][local_idx])
        coverage_pred = int(pred["coverage_pred"][local_idx])
        out.append(
            {
                "split": split,
                "target_idea_id": row.get("target_idea_id"),
                "prior_idea_id": row.get("prior_idea_id"),
                "target_date": row.get("target_date"),
                "prior_date": row.get("prior_date"),
                "gold_pair_label": row.get("coverage_label"),
                "gold_coverage_score": safe_float(row.get("coverage_score")),
                "gold_relatedness": "related" if relatedness(row) else "unrelated",
                "pred_pair_label": pair_labels[pair_pred],
                "pred_pair_probabilities": {
                    pair_labels[idx]: round(float(value), 6)
                    for idx, value in enumerate(pred["pair_prob"][local_idx].tolist())
                },
                "pred_relatedness_probability": round(float(pred["related_prob"][local_idx]), 6),
                "pred_covering_probability_given_related": round(float(pred["covering_prob"][local_idx]), 6),
                "pred_large_probability_given_covering": round(float(pred["strength_prob"][local_idx]), 6),
                "pred_coverage_level": COVERAGE_LEVELS[coverage_pred],
                "pred_coverage_level_probabilities": {
                    COVERAGE_LEVELS[idx]: round(float(value), 6)
                    for idx, value in enumerate(pred["coverage_prob"][local_idx].tolist())
                },
                "pred_coverage_score": round(float(exported_score[local_idx]), 6),
                "pred_score_head_coverage_score": round(float(pred["score_pred"][local_idx]), 6),
                "pred_pair_expected_score": round(float(pred["pair_expected_score"][local_idx]), 6),
                "pred_coverage_score_source": score_source,
            }
        )
    return out


def sampler_weights(
    rows: list[dict[str, Any]],
    indices: list[int],
    pair_labels: list[str],
    strategy: str,
    label_alpha: float,
    target_beta: float,
) -> torch.Tensor | None:
    if strategy == "none":
        return None
    pair_ids = [pair_label_id(rows[idx], pair_labels) for idx in indices]
    label_alpha = min(1.0, max(0.0, float(label_alpha)))
    target_beta = min(1.0, max(0.0, float(target_beta)))
    if strategy in {"label", "sqrt_label"}:
        counts = Counter(pair_ids)
        exponent = 0.5 if strategy == "sqrt_label" else label_alpha
        weights = [1.0 / (float(counts[label_id]) ** exponent) for label_id in pair_ids]
    elif strategy == "covering":
        groups = [0 if label_id == 0 else 1 if label_id == 1 else 2 for label_id in pair_ids]
        counts = Counter(groups)
        weights = [1.0 / float(counts[group]) for group in groups]
    else:
        raise ValueError(f"unknown sampler strategy: {strategy}")
    if target_beta > 0:
        target_ids = [str(rows[idx].get("target_idea_id") or f"row:{idx}") for idx in indices]
        target_counts = Counter(target_ids)
        weights = [
            weight * (1.0 / (float(target_counts[target_id]) ** target_beta))
            for weight, target_id in zip(weights, target_ids, strict=True)
        ]
    return torch.tensor(weights, dtype=torch.double)


def build_loader(
    rows: list[dict[str, Any]],
    indices: list[int],
    pair_labels: list[str],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    label_sample_weights: dict[str, float] | None = None,
    sampler_strategy: str = "none",
    sampler_alpha: float = 1.0,
    target_sampler_beta: float = 0.0,
) -> DataLoader[dict[str, Any]]:
    dataset = PairCoverageDataset(rows, indices, pair_labels, label_sample_weights)
    collator = PairCoverageCollator(tokenizer, max_length)
    weights = (
        sampler_weights(rows, indices, pair_labels, sampler_strategy, sampler_alpha, target_sampler_beta)
        if shuffle
        else None
    )
    sampler = None
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_rows_and_splits(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int], list[int], list[int], str]:
    explicit_files = [args.train_pair_file, args.dev_pair_file, args.test_pair_file]
    if any(explicit_files):
        if not all(explicit_files):
            raise ValueError("--train-pair-file, --dev-pair-file, and --test-pair-file must be provided together")
        rows: list[dict[str, Any]] = []
        split_indices: dict[str, list[int]] = {}
        for split, path in [
            ("train", args.train_pair_file),
            ("dev", args.dev_pair_file),
            ("test", args.test_pair_file),
        ]:
            split_rows = [
                dict(row, split=split)
                for row in read_jsonl(path)
                if str(row.get("coverage_label") or "") in COVERAGE_TO_LEVEL
            ]
            start = len(rows)
            rows.extend(split_rows)
            split_indices[split] = list(range(start, len(rows)))
        return rows, split_indices["train"], split_indices["dev"], split_indices["test"], "explicit_files"

    rows = [
        row
        for row in read_jsonl(args.pair_file)
        if str(row.get("coverage_label") or "") in COVERAGE_TO_LEVEL
    ]
    if args.use_split_field and rows and all(str(row.get("split") or "") in {"train", "dev", "test"} for row in rows):
        return (
            rows,
            [idx for idx, row in enumerate(rows) if row.get("split") == "train"],
            [idx for idx, row in enumerate(rows) if row.get("split") == "dev"],
            [idx for idx, row in enumerate(rows) if row.get("split") == "test"],
            "split_field",
        )

    train_idx, dev_idx, test_idx = temporal_split(rows, "target_date")
    return rows, train_idx, dev_idx, test_idx, "temporal_split"


def save_checkpoint(
    model: PairCoverageModel,
    tokenizer: Any,
    output_dir: Path,
    cfg: PairCoverageConfig,
    metrics: dict[str, Any],
    pair_labels: list[str],
) -> None:
    path = output_dir / "best"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
            "coverage_levels": COVERAGE_LEVELS,
            "pair_labels": pair_labels,
        },
        path / "checkpoint.pt",
    )
    tokenizer.save_pretrained(path / "tokenizer")
    (path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "pair_macro_f1":
        return float(metrics["pair_combined"]["macro_f1"])
    if name == "pair_qwk":
        return float(metrics["pair_combined"]["qwk"])
    if name == "pair_covering_f1":
        return float(metrics["pair_combined"].get("covering_binary", {}).get("f1", 0.0))
    if name == "pair_covering_precision":
        return float(metrics["pair_combined"].get("covering_binary", {}).get("precision", 0.0))
    if name == "pair_covering_recall":
        return float(metrics["pair_combined"].get("covering_binary", {}).get("recall", 0.0))
    if name == "pair_covering_balanced":
        covering = metrics["pair_combined"].get("covering_binary", {})
        return (
            0.45 * float(covering.get("f1", 0.0))
            + 0.30 * float(metrics["pair_combined"]["qwk"])
            + 0.25 * float(metrics["pair_combined"]["macro_f1"])
        )
    if name == "pair_ranking_balanced":
        covering = metrics["pair_combined"].get("covering_binary", {})
        return (
            0.35 * float(metrics["ranking"]["pairwise_accuracy"])
            + 0.25 * float(metrics["pair_combined"]["qwk"])
            + 0.25 * float(covering.get("f1", 0.0))
            + 0.15 * float(metrics["pair_combined"]["macro_f1"])
        )
    if name == "coverage_qwk":
        return float(metrics["coverage_ordinal"]["qwk"])
    if name == "ranking_pairwise_accuracy":
        return float(metrics["ranking"]["pairwise_accuracy"])
    return float(metrics["pair_combined"]["macro_f1"])


def parse_label_sample_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("--label-sample-weights entries must use label=value")
        label, value = item.split("=", 1)
        label = label.strip()
        if label == "full_cover":
            label = "large_cover"
        if label not in BASE_PAIR_LABELS:
            raise ValueError(f"Unknown pair label in --label-sample-weights: {label}")
        weights[label] = max(0.0, float(value.strip()))
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, default=Path("dataset/idea_pair_coverage_llm.jsonl"))
    parser.add_argument("--train-pair-file", type=Path, default=None)
    parser.add_argument("--dev-pair-file", type=Path, default=None)
    parser.add_argument("--test-pair-file", type=Path, default=None)
    parser.add_argument("--use-split-field", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/stage_pair_coverage"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-schema", choices=["staged", "ordinal"], default="staged")
    parser.add_argument(
        "--hierarchical-only",
        action="store_true",
        help="Use only the relatedness/covering/strength binary hierarchy; disables direct class and score losses/blending.",
    )
    parser.add_argument("--coverage-loss-weight", type=float, default=1.0)
    parser.add_argument("--relatedness-loss-weight", type=float, default=0.4)
    parser.add_argument("--covering-loss-weight", type=float, default=0.8)
    parser.add_argument("--strength-loss-weight", type=float, default=0.6)
    parser.add_argument("--direct-loss-weight", type=float, default=0.7)
    parser.add_argument("--score-loss-weight", type=float, default=0.3)
    parser.add_argument(
        "--pair-expected-loss-weight",
        type=float,
        default=0.0,
        help="Optional MSE loss on hierarchical expected evidence score: not/related=0, partial=0.5, large=1.0.",
    )
    parser.add_argument("--direct-blend-weight", type=float, default=0.35)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument(
        "--sampler",
        choices=["none", "label", "sqrt_label", "covering"],
        default="none",
        help="Optional train-time weighted sampler; label uses --sampler-alpha, sqrt_label is alpha=0.5.",
    )
    parser.add_argument(
        "--sampler-alpha",
        type=float,
        default=1.0,
        help="Exponent for --sampler label. 1.0 fully balances labels; 0.5 is square-root balancing.",
    )
    parser.add_argument(
        "--target-sampler-beta",
        type=float,
        default=0.0,
        help="Optional target-level sampling exponent to reduce domination by targets with many pairs.",
    )
    parser.add_argument(
        "--export-score-source",
        choices=["auto", "score_head", "pair_expected"],
        default="auto",
        help="Which score to write as pred_coverage_score for calibration/ranking.",
    )
    parser.add_argument(
        "--pair-expected-score-mode",
        choices=["auto", "label_mean", "coverage_evidence"],
        default="auto",
        help="How pair probabilities are converted to pair_expected score.",
    )
    parser.add_argument("--ranking-margin", type=float, default=0.15)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "pair_macro_f1",
            "pair_qwk",
            "pair_covering_f1",
            "pair_covering_precision",
            "pair_covering_recall",
            "pair_covering_balanced",
            "pair_ranking_balanced",
            "coverage_qwk",
            "ranking_pairwise_accuracy",
        ],
        default="pair_qwk",
    )
    parser.add_argument(
        "--label-sample-weights",
        default="",
        help="Optional comma-separated label=weight overrides, e.g. related_not_covering=1.5,partial_cover=1.2",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.hierarchical_only:
        args.label_schema = "staged"
        args.direct_loss_weight = 0.0
        args.score_loss_weight = 0.0
        args.direct_blend_weight = 0.0
    effective_score_source = args.export_score_source
    if effective_score_source == "auto":
        effective_score_source = "pair_expected" if args.hierarchical_only else "score_head"
    effective_pair_score_mode = args.pair_expected_score_mode
    if effective_pair_score_mode == "auto":
        effective_pair_score_mode = "coverage_evidence" if args.hierarchical_only else "label_mean"

    set_seed(args.seed)
    rows, train_idx, dev_idx, test_idx, split_source = load_rows_and_splits(args)
    pair_labels = BASE_PAIR_LABELS if args.label_schema == "staged" else pair_label_names(rows)
    label_sample_weights = parse_label_sample_weights(args.label_sample_weights)
    rng = random.Random(args.seed)
    if args.max_train_samples and len(train_idx) > args.max_train_samples:
        by_label: dict[int, list[int]] = defaultdict(list)
        for idx in train_idx:
            by_label[pair_label_id(rows[idx], pair_labels)].append(idx)
        sampled: list[int] = []
        per_class = max(1, args.max_train_samples // max(1, len(by_label)))
        for values in by_label.values():
            rng.shuffle(values)
            sampled.extend(values[:per_class])
        if len(sampled) < args.max_train_samples:
            seen = set(sampled)
            rest = [idx for idx in train_idx if idx not in seen]
            rng.shuffle(rest)
            sampled.extend(rest[: args.max_train_samples - len(sampled)])
        train_idx = sorted(sampled, key=lambda idx: (int(rows[idx].get("target_date") or 0), idx))
    if args.max_eval_samples:
        dev_idx = dev_idx[: args.max_eval_samples]
        test_idx = test_idx[: args.max_eval_samples]

    device = device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    cfg = PairCoverageConfig(
        model_name=args.model_name,
        dropout=args.dropout,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model = PairCoverageModel(cfg).to(device)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    train_loader = build_loader(
        rows,
        train_idx,
        pair_labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        True,
        args.num_workers,
        label_sample_weights,
        args.sampler,
        args.sampler_alpha,
        args.target_sampler_beta,
    )
    dev_loader = build_loader(rows, dev_idx, pair_labels, tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers)
    test_loader = build_loader(rows, test_idx, pair_labels, tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers)

    coverage_pos_weight, related_pos_weight = compute_pos_weights(rows, train_idx, device, args.pos_weight_cap, args.pos_weight_scale)
    staged_weights = compute_staged_weights(rows, train_idx, pair_labels, device, args.pos_weight_cap, args.pos_weight_scale)
    pair_score_levels = pair_score_levels_from_rows(rows, train_idx, pair_labels, effective_pair_score_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(0.01, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": asdict(cfg),
        "split_source": split_source,
        "coverage_levels": COVERAGE_LEVELS,
        "pair_labels": pair_labels,
        "split_sizes": {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)},
        "class_distribution": dict(Counter(str(row.get("coverage_label") or "") for row in rows)),
        "relatedness_distribution": dict(Counter("related" if relatedness(row) else "unrelated" for row in rows)),
        "label_sample_weights": label_sample_weights,
        "effective_score_source": effective_score_source,
        "effective_pair_expected_score_mode": effective_pair_score_mode,
        "pair_score_levels": {
            label: round(float(pair_score_levels[idx]), 6)
            for idx, label in enumerate(pair_labels)
        },
        "loss_pos_weights": {
            "coverage_thresholds": [round(float(x), 4) for x in coverage_pos_weight.detach().cpu().tolist()],
            "relatedness": round(float(related_pos_weight.detach().cpu()), 4),
            "staged_relatedness": round(float(staged_weights["related"].detach().cpu()), 4),
            "staged_covering": round(float(staged_weights["covering"].detach().cpu()), 4),
            "staged_strength": round(float(staged_weights["strength"].detach().cpu()), 4),
            "direct_pair": [round(float(x), 4) for x in staged_weights["direct"].detach().cpu().tolist()],
        },
        "leakage_control": [
            "the direct target-prior relation field is never added to the prompt",
            "gold coverage labels/scores are used only as losses and metrics",
            "temporal split is based on target_date",
        ],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -float("inf")
    best_dev_metrics: dict[str, Any] | None = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, list[float]] = defaultdict(list)
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            labels = {key: value.to(device) for key, value in batch["labels"].items()}
            inputs = move_to_device(batch["inputs"], device)
            with amp_context(device, args.amp, args.bf16):
                outputs = model(inputs)
                if args.label_schema == "staged":
                    loss, parts = compute_staged_loss(
                        outputs,
                        labels,
                        staged_weights,
                        args.relatedness_loss_weight,
                        args.covering_loss_weight,
                        args.strength_loss_weight,
                        args.direct_loss_weight,
                        args.score_loss_weight,
                        args.pair_expected_loss_weight,
                    )
                else:
                    loss, parts = compute_loss(
                        outputs,
                        labels,
                        coverage_pos_weight,
                        related_pos_weight,
                        args.coverage_loss_weight,
                        args.relatedness_loss_weight,
                        args.score_loss_weight,
                    )
            if not torch.isfinite(loss).all():
                optimizer.zero_grad(set_to_none=True)
                print(json.dumps({"warning": "non_finite_loss", "epoch": epoch, "step": step, "loss_parts": parts}, ensure_ascii=False))
                continue
            loss.backward()
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
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        dev_pred = predict(
            model,
            dev_loader,
            device,
            pair_labels,
            pair_score_levels,
            args.label_schema,
            args.direct_blend_weight,
            args.amp,
            args.bf16,
        )
        dev_metrics = split_metrics(rows, dev_idx, dev_pred, pair_labels, args.ranking_margin, effective_score_source)
        dev_metrics["epoch"] = epoch
        dev_metrics["global_step"] = global_step
        dev_metrics["loss"] = {name: round(float(np.mean(values)), 6) for name, values in running.items()}
        print(json.dumps({"split": "dev", **dev_metrics}, ensure_ascii=False, indent=2))
        (args.output_dir / f"dev_epoch_{epoch}.json").write_text(json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        score = metric_value(dev_metrics, args.selection_metric)
        if score > best_score:
            best_score = score
            best_dev_metrics = dev_metrics
            save_checkpoint(model, tokenizer, args.output_dir, cfg, dev_metrics, pair_labels)

    best_path = args.output_dir / "best" / "checkpoint.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    all_predictions: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {"best_dev": best_dev_metrics}
    for split_name, split_idx, loader in [
        ("train", train_idx, build_loader(rows, train_idx, pair_labels, tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers)),
        ("dev", dev_idx, dev_loader),
        ("test", test_idx, test_loader),
    ]:
        pred = predict(
            model,
            loader,
            device,
            pair_labels,
            pair_score_levels,
            args.label_schema,
            args.direct_blend_weight,
            args.amp,
            args.bf16,
        )
        metrics = split_metrics(rows, split_idx, pred, pair_labels, args.ranking_margin, effective_score_source)
        final_metrics[split_name] = metrics
        split_rows = prediction_rows(rows, split_name, pred, pair_labels, effective_score_source)
        write_jsonl(args.output_dir / "predictions" / f"{split_name}.jsonl", split_rows)
        all_predictions.extend(split_rows)
        print(json.dumps({"split": split_name, **metrics}, ensure_ascii=False, indent=2))
    write_jsonl(args.output_dir / "predictions" / "all.jsonl", all_predictions)
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
