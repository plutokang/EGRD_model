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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


JOINT_LABELS = ["not_covered", "weakly_covered", "partially_covered", "mostly_covered"]
JOINT_TO_ID = {label: idx for idx, label in enumerate(JOINT_LABELS)}


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def temporal_split(rows: list[dict[str, Any]], novelty_index: dict[str, dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    dated: list[tuple[int, int]] = []
    for idx, row in enumerate(rows):
        target_id = str(row.get("target_idea_id") or "")
        novelty = novelty_index.get(target_id, {})
        date = int(row.get("target_date") or novelty.get("target_date") or 0)
        if date > 0:
            dated.append((idx, date))
    dated.sort(key=lambda item: (item[1], item[0]))
    n = len(dated)
    return (
        [idx for idx, _ in dated[: int(n * 0.70)]],
        [idx for idx, _ in dated[int(n * 0.70): int(n * 0.85)]],
        [idx for idx, _ in dated[int(n * 0.85):]],
    )


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    observed = np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.float64)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(1.0, observed.sum())
    weights = np.zeros_like(observed)
    denom = max(1, (n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / denom
    den = float((weights * expected).sum())
    return float(1 - (weights * observed).sum() / den) if den else 0.0


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out = {}
    for idx, label in enumerate(JOINT_LABELS):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int((y_true == idx).sum()),
        }
    return out


def classification_metrics(y_true: list[int], y_pred: list[int], proba: list[list[float]]) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(proba, dtype=np.float32)
    per_class = per_class_metrics(true, pred)
    f1s = [item["f1"] for item in per_class.values()]
    over_error = np.maximum(0, pred - true)
    under_error = np.maximum(0, true - pred)
    return {
        "accuracy": round(float((true == pred).mean()), 4) if len(true) else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "ordinal_mae": round(float(np.abs(true - pred).mean()), 4) if len(true) else 0.0,
        "qwk": round(qwk(true, pred, len(JOINT_LABELS)), 4) if len(true) else 0.0,
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4) if len(probs) else 0.0,
        "mean_over_error": round(float(over_error.mean()), 4) if len(true) else 0.0,
        "mean_under_error": round(float(under_error.mean()), 4) if len(true) else 0.0,
        "not_covered_recall": per_class["not_covered"]["recall"],
        "mostly_precision": per_class["mostly_covered"]["precision"],
        "mostly_recall": per_class["mostly_covered"]["recall"],
        "per_class": per_class,
        "pred_distribution": dict(Counter(JOINT_LABELS[int(idx)] for idx in pred)),
        "true_distribution": dict(Counter(JOINT_LABELS[int(idx)] for idx in true)),
    }


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    if len(true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0}
    err = pred - true
    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denom = float(np.sqrt((true_centered**2).sum()) * np.sqrt((pred_centered**2).sum()))
    pearson = float((true_centered * pred_centered).sum() / denom) if denom else 0.0
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err**2).mean())), 4),
        "pearson": round(pearson, 4),
    }


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(len(JOINT_LABELS) - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def ordinal_class_probs(ordinal_logits: torch.Tensor) -> torch.Tensor:
    gt = torch.sigmoid(ordinal_logits.float())
    gt = torch.cummin(gt, dim=1).values
    p0 = 1.0 - gt[:, 0]
    p1 = (gt[:, 0] - gt[:, 1]).clamp_min(0.0)
    p2 = (gt[:, 1] - gt[:, 2]).clamp_min(0.0)
    p3 = gt[:, 2]
    probs = torch.stack([p0, p1, p2, p3], dim=-1)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def load_novelty_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("target_idea_id") or ""): row
        for row in read_jsonl(path)
        if row.get("target_idea_id")
    }


@dataclass
class TextLookup:
    target_prior_text: dict[tuple[str, str], str]
    idea_text: dict[str, str]
    target_prior_meta: dict[tuple[str, str], dict[str, Any]]


def build_text_lookup(novelty_index: dict[str, dict[str, Any]], pair_text_files: list[Path]) -> TextLookup:
    target_prior_text: dict[tuple[str, str], str] = {}
    target_prior_meta: dict[tuple[str, str], dict[str, Any]] = {}
    idea_text: dict[str, str] = {}

    for target_id, row in novelty_index.items():
        target_text = str(row.get("target_text") or "")
        if target_text:
            idea_text[target_id] = target_text
        prior_ids = list(row.get("top_priors") or row.get("pre_recomputed_top_priors") or [])
        prior_texts = list(row.get("top_prior_texts") or [])
        prior_labels = list(row.get("top_prior_coverage_labels") or [])
        prior_scores = list(row.get("top_prior_coverage_scores") or [])
        for idx, prior_id_raw in enumerate(prior_ids):
            prior_id = str(prior_id_raw or "")
            if not prior_id:
                continue
            if idx < len(prior_texts) and prior_texts[idx]:
                text = str(prior_texts[idx])
                target_prior_text.setdefault((target_id, prior_id), text)
                idea_text.setdefault(prior_id, text)
            meta: dict[str, Any] = {}
            if idx < len(prior_labels):
                meta["coverage_label"] = prior_labels[idx]
            if idx < len(prior_scores):
                meta["coverage_score"] = prior_scores[idx]
            if meta:
                target_prior_meta.setdefault((target_id, prior_id), meta)

    for path in pair_text_files:
        if not path:
            continue
        for row in read_jsonl(path):
            target_id = str(row.get("target_idea_id") or "")
            prior_id = str(row.get("prior_idea_id") or "")
            if not target_id or not prior_id:
                continue
            target_text = str(row.get("target_text") or "")
            prior_text = str(row.get("prior_text") or "")
            if target_text:
                idea_text.setdefault(target_id, target_text)
            if prior_text:
                target_prior_text[(target_id, prior_id)] = prior_text
                idea_text.setdefault(prior_id, prior_text)
            target_prior_meta.setdefault((target_id, prior_id), {}).update(
                {
                    key: row.get(key)
                    for key in [
                        "prior_date",
                        "prior_primary_aspect",
                        "prior_contribution_type",
                        "prior_task_family",
                        "coverage_label",
                        "coverage_score",
                    ]
                    if row.get(key) is not None
                }
            )
    return TextLookup(target_prior_text, idea_text, target_prior_meta)


def load_ranking_splits(
    args: argparse.Namespace,
    novelty_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if args.ranking_dir is not None:
        return (
            {
                split: read_jsonl(args.ranking_dir / f"{split}.jsonl")
                for split in ["train", "dev", "test"]
            },
            "ranking_dir",
        )

    explicit = [args.train_ranking_file, args.dev_ranking_file, args.test_ranking_file]
    if any(explicit):
        if not all(explicit):
            raise ValueError("--train-ranking-file, --dev-ranking-file, and --test-ranking-file must be provided together")
        return (
            {
                "train": read_jsonl(args.train_ranking_file),
                "dev": read_jsonl(args.dev_ranking_file),
                "test": read_jsonl(args.test_ranking_file),
            },
            "explicit_ranking_files",
        )

    if args.ranking_file is not None:
        rows = read_jsonl(args.ranking_file)
    else:
        rows = [
            {
                "split": row.get("split"),
                "target_idea_id": row.get("target_idea_id"),
                "target_text": row.get("target_text"),
                "target_date": row.get("target_date"),
                "candidate_priors": [
                    {
                        "rank": idx + 1,
                        "prior_idea_id": prior_id,
                        "prior_text": prior_texts[idx] if idx < len(prior_texts) else "",
                        "coverage_label": prior_labels[idx] if idx < len(prior_labels) else "",
                        "coverage_score": prior_scores[idx] if idx < len(prior_scores) else 0.0,
                    }
                    for idx, prior_id in enumerate(list(row.get("top_priors") or []))
                ],
            }
            for row in novelty_index.values()
            for prior_texts, prior_labels, prior_scores in [
                (
                    list(row.get("top_prior_texts") or []),
                    list(row.get("top_prior_coverage_labels") or []),
                    list(row.get("top_prior_coverage_scores") or []),
                )
            ]
        ]

    valid_rows = [row for row in rows if row.get("target_idea_id") in novelty_index]
    if valid_rows and all(str(row.get("split") or "") in {"train", "dev", "test"} for row in valid_rows):
        return (
            {
                "train": [row for row in valid_rows if row.get("split") == "train"],
                "dev": [row for row in valid_rows if row.get("split") == "dev"],
                "test": [row for row in valid_rows if row.get("split") == "test"],
            },
            "split_field",
        )

    train_idx, dev_idx, test_idx = temporal_split(valid_rows, novelty_index)
    return (
        {
            "train": [valid_rows[idx] for idx in train_idx],
            "dev": [valid_rows[idx] for idx in dev_idx],
            "test": [valid_rows[idx] for idx in test_idx],
        },
        "temporal_split",
    )


def ranking_priors(row: dict[str, Any]) -> list[dict[str, Any]]:
    priors = row.get("ranked_priors")
    if isinstance(priors, list):
        return [dict(item) for item in priors]
    priors = row.get("candidate_priors")
    if isinstance(priors, list):
        return [dict(item) for item in priors]
    return []


def prior_signal(prior: dict[str, Any], meta: dict[str, Any], key_options: list[str], default: Any = "") -> Any:
    for key in key_options:
        if prior.get(key) is not None:
            return prior.get(key)
    for key in key_options:
        if meta.get(key) is not None:
            return meta.get(key)
    return default


def prior_text(target_id: str, prior: dict[str, Any], text_lookup: TextLookup) -> str:
    direct = str(prior.get("prior_text") or "")
    if direct:
        return direct
    prior_id = str(prior.get("prior_idea_id") or "")
    return text_lookup.target_prior_text.get((target_id, prior_id)) or text_lookup.idea_text.get(prior_id, "")


def joint_union_score(scores: list[float]) -> float:
    value = 1.0
    for score in scores:
        value *= 1.0 - min(0.99, max(0.0, score))
    return 1.0 - value


def coverage_level_from_label(label: Any) -> float:
    label_text = str(label or "")
    if label_text in {"large_cover", "mostly_covered"}:
        return 1.0
    if label_text in {"partial_cover", "partially_covered"}:
        return 2.0 / 3.0
    if label_text in {"related_not_covering", "weakly_covered"}:
        return 1.0 / 3.0
    return 0.0


def metadata_line(row: dict[str, Any], novelty: dict[str, Any]) -> str:
    fields = [
        f"aspect={novelty.get('primary_aspect') or row.get('target_primary_aspect') or 'unknown'}",
        f"contribution={novelty.get('contribution_type') or row.get('target_contribution_type') or 'unknown'}",
        f"task={novelty.get('task_family') or row.get('target_task_family') or 'unknown'}",
        f"date={novelty.get('target_date') or row.get('target_date') or 'unknown'}",
    ]
    return "; ".join(fields)


def prompt_from_example(
    row: dict[str, Any],
    novelty: dict[str, Any],
    text_lookup: TextLookup,
    top_k: int,
    include_prior_signals: bool,
    include_summary_features: bool,
) -> tuple[str, list[dict[str, Any]]]:
    target_id = str(row.get("target_idea_id") or "")
    target_text = str(row.get("target_text") or novelty.get("target_text") or text_lookup.idea_text.get(target_id, ""))
    priors = ranking_priors(row)[:top_k]
    prompt_priors = []
    scores = []
    label_levels = []

    lines = [
        "Predict how much the historical priors jointly cover the target scientific idea.",
        "Use the target text, prior texts, and pair-level signals. Do not treat relatedness alone as coverage.",
        "Joint labels: not_covered, weakly_covered, partially_covered, mostly_covered.",
        f"[TARGET METADATA] {metadata_line(row, novelty)}",
        f"[TARGET IDEA]\n{target_text}",
        "[HISTORICAL PRIORS]",
    ]
    for idx, prior in enumerate(priors, start=1):
        prior_id = str(prior.get("prior_idea_id") or "")
        meta = text_lookup.target_prior_meta.get((target_id, prior_id), {})
        text = prior_text(target_id, prior, text_lookup)
        label = prior_signal(prior, meta, ["calibrated_pair_label", "coverage_label", "pred_pair_label", "gold_pair_label"], "unknown")
        score = safe_float(prior_signal(prior, meta, ["calibrated_score", "coverage_score", "pred_coverage_score", "gold_coverage_score"], 0.0), 0.0)
        rank_score = safe_float(prior.get("rank_score"), score)
        scores.append(score)
        label_levels.append(coverage_level_from_label(label))
        signal_text = ""
        if include_prior_signals:
            signal_text = f" pair_label={label}; pair_score={score:.3f}; rank_score={rank_score:.3f};"
        prior_meta = []
        for key, label_name in [
            ("prior_date", "date"),
            ("prior_primary_aspect", "aspect"),
            ("prior_contribution_type", "contribution"),
            ("prior_task_family", "task"),
        ]:
            if meta.get(key) is not None:
                prior_meta.append(f"{label_name}={meta[key]}")
        meta_text = f" {'; '.join(prior_meta)};" if prior_meta else ""
        lines.append(f"[PRIOR {idx} id={prior_id};{signal_text}{meta_text}]\n{text}")
        prompt_priors.append(
            {
                "rank": idx,
                "prior_idea_id": prior_id,
                "prior_text_found": bool(text),
                "pair_label": label,
                "pair_score": round(float(score), 6),
                "rank_score": round(float(rank_score), 6),
            }
        )

    if include_summary_features:
        covering_scores = [score for score, level in zip(scores, label_levels, strict=False) if level >= 2.0 / 3.0]
        summary = [
            f"num_priors={len(priors)}",
            f"max_pair_score={(max(scores) if scores else 0.0):.3f}",
            f"mean_pair_score={(float(np.mean(scores)) if scores else 0.0):.3f}",
            f"union_pair_score={joint_union_score(scores):.3f}",
            f"covering_prior_ratio={(len(covering_scores) / max(1, len(priors))):.3f}",
        ]
        lines.append(f"[PAIR SIGNAL SUMMARY] {'; '.join(summary)}")
    return "\n".join(lines), prompt_priors


class JointTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        novelty_index: dict[str, dict[str, Any]],
        text_lookup: TextLookup,
        top_k: int,
        include_prior_signals: bool,
        include_summary_features: bool,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        missing_text = 0
        for row in rows:
            target_id = str(row.get("target_idea_id") or "")
            novelty = novelty_index.get(target_id)
            if not novelty:
                continue
            label = str(novelty.get("joint_coverage_label") or "")
            if label not in JOINT_TO_ID:
                continue
            prompt, prompt_priors = prompt_from_example(
                row,
                novelty,
                text_lookup,
                top_k,
                include_prior_signals,
                include_summary_features,
            )
            missing_text += sum(1 for item in prompt_priors if not item["prior_text_found"])
            self.items.append(
                {
                    "target_idea_id": target_id,
                    "prompt": prompt,
                    "prompt_priors": prompt_priors,
                    "ranking": row,
                    "label": JOINT_TO_ID[label],
                    "score": safe_float(novelty.get("joint_coverage_score"), 0.0),
                    "weight": min(1.0, max(0.2, safe_float(novelty.get("label_confidence"), 1.0))),
                }
            )
        if not self.items:
            raise ValueError("No joinable joint coverage rows")
        self.missing_prior_texts = missing_text

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class JointTextCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["prompt"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "inputs": encoded,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "score": torch.tensor([item["score"] for item in batch], dtype=torch.float32),
            "weight": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
            "target_idea_id": [item["target_idea_id"] for item in batch],
            "ranking": [item["ranking"] for item in batch],
            "prompt_priors": [item["prompt_priors"] for item in batch],
        }


@dataclass
class JointTextConfig:
    model_name: str
    dropout: float = 0.1
    trust_remote_code: bool = True
    cache_dir: str | None = None
    local_files_only: bool = False


class JointTextModel(nn.Module):
    def __init__(self, cfg: JointTextConfig) -> None:
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
        self.class_head = nn.Linear(hidden_size, len(JOINT_LABELS))
        self.ordinal_head = nn.Linear(hidden_size, len(JOINT_LABELS) - 1)
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
            "logits": self.class_head(h),
            "ordinal_logits": self.ordinal_head(h),
            "score": torch.sigmoid(self.score_head(h).squeeze(-1)),
        }


def move_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def class_weights(dataset: JointTextDataset, device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in dataset.items)
    total = sum(counts.values())
    weights = []
    for idx in range(len(JOINT_LABELS)):
        value = total / max(1, len(JOINT_LABELS) * counts.get(idx, 0))
        weights.append(min(5.0, max(0.1, math.sqrt(value))))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def ordinal_pos_weights(dataset: JointTextDataset, device: torch.device) -> torch.Tensor:
    labels = np.asarray([int(row["label"]) for row in dataset.items], dtype=np.int64)
    targets = np.stack([(labels > threshold).astype(np.float32) for threshold in range(len(JOINT_LABELS) - 1)], axis=1)
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = np.ones(len(JOINT_LABELS) - 1, dtype=np.float32)
    for idx in range(len(weights)):
        if pos[idx] > 0:
            weights[idx] = min(10.0, max(0.1, float(neg[idx] / pos[idx])))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    ce_weight: float,
    ordinal_weight: float,
    score_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["label"].to(outputs["logits"].device)
    scores = batch["score"].to(outputs["logits"].device)
    sample_weight = batch["weight"].to(outputs["logits"].device)
    ce = F.cross_entropy(outputs["logits"], labels, weight=weights, reduction="none")
    ce = (ce * sample_weight).mean()
    ordinal = F.binary_cross_entropy_with_logits(
        outputs["ordinal_logits"],
        ordinal_targets(labels),
        pos_weight=ordinal_weights,
        reduction="none",
    ).mean(dim=-1)
    ordinal = (ordinal * sample_weight).mean()
    score_loss = F.mse_loss(outputs["score"], scores, reduction="none")
    score_loss = (score_loss * sample_weight).mean()
    total = ce_weight * ce + ordinal_weight * ordinal + score_weight * score_loss
    return total, {
        "total": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "ordinal": float(ordinal.detach().cpu()),
        "score": float(score_loss.detach().cpu()),
    }


def probabilities(outputs: dict[str, torch.Tensor], prediction_mode: str, ordinal_blend_weight: float) -> torch.Tensor:
    class_probs = torch.softmax(outputs["logits"].float(), dim=-1)
    if prediction_mode == "class":
        return class_probs
    ordinal_probs = ordinal_class_probs(outputs["ordinal_logits"])
    if prediction_mode == "ordinal":
        return ordinal_probs
    weight = min(1.0, max(0.0, float(ordinal_blend_weight)))
    probs = (1.0 - weight) * class_probs + weight * ordinal_probs
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


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
def evaluate(
    model: JointTextModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    args: argparse.Namespace,
    return_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prob_all: list[list[float]] = []
    score_true: list[float] = []
    score_pred: list[float] = []
    losses: dict[str, list[float]] = defaultdict(list)
    pred_rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = move_to_device(batch["inputs"], device)
        with amp_context(device, args.amp, args.bf16):
            outputs = model(inputs)
            loss, parts = compute_loss(
                outputs,
                batch,
                weights,
                ordinal_weights,
                args.ce_loss_weight,
                args.ordinal_loss_weight,
                args.score_loss_weight,
            )
        probs = probabilities(outputs, args.prediction_mode, args.ordinal_blend_weight)
        pred = probs.argmax(dim=-1).detach().cpu().numpy()
        labels = batch["label"].numpy()
        scores = batch["score"].numpy()
        pred_scores = outputs["score"].detach().float().cpu().numpy()
        y_true.extend(int(x) for x in labels)
        y_pred.extend(int(x) for x in pred)
        prob_batch = probs.detach().float().cpu().numpy().tolist()
        prob_all.extend(prob_batch)
        score_true.extend(float(x) for x in scores)
        score_pred.extend(float(x) for x in pred_scores)
        for name, value in parts.items():
            losses[name].append(value)
        losses["loss"].append(float(loss.detach().cpu()))
        if return_predictions:
            for idx, target_id in enumerate(batch["target_idea_id"]):
                pred_rows.append(
                    {
                        "target_idea_id": target_id,
                        "gold_joint_coverage_label": JOINT_LABELS[int(labels[idx])],
                        "gold_joint_coverage_score": round(float(scores[idx]), 6),
                        "pred_joint_coverage_label": JOINT_LABELS[int(pred[idx])],
                        "pred_joint_coverage_score": round(float(pred_scores[idx]), 6),
                        "pred_joint_probabilities": {
                            label: round(float(prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "ranked_priors": batch["ranking"][idx].get("ranked_priors")
                        or batch["ranking"][idx].get("candidate_priors"),
                        "prompt_priors": batch["prompt_priors"][idx],
                    }
                )
    metrics = classification_metrics(y_true, y_pred, prob_all)
    metrics["score_metrics"] = regression_metrics(score_true, score_pred)
    metrics["loss"] = {name: round(float(np.mean(values)), 6) for name, values in losses.items()}
    if return_predictions:
        metrics["_predictions"] = pred_rows
    return metrics


def metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "neg_ordinal_mae":
        return -float(metrics["ordinal_mae"])
    if name == "score_pearson":
        return float(metrics["score_metrics"]["pearson"])
    return float(metrics[name])


def build_loader(
    dataset: JointTextDataset,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=JointTextCollator(tokenizer, max_length),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def sample_train_rows(rows: list[dict[str, Any]], novelty_index: dict[str, dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        novelty = novelty_index.get(str(row.get("target_idea_id") or ""))
        if not novelty:
            continue
        label = JOINT_TO_ID.get(str(novelty.get("joint_coverage_label") or ""), 0)
        by_label[label].append(row)
    sampled: list[dict[str, Any]] = []
    per_class = max(1, max_samples // max(1, len(by_label)))
    for values in by_label.values():
        rng.shuffle(values)
        sampled.extend(values[:per_class])
    if len(sampled) < max_samples:
        seen = {id(row) for row in sampled}
        rest = [row for row in rows if id(row) not in seen]
        rng.shuffle(rest)
        sampled.extend(rest[: max_samples - len(sampled)])
    return sampled[:max_samples]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-dir", type=Path, default=None)
    parser.add_argument("--ranking-file", type=Path, default=None)
    parser.add_argument("--train-ranking-file", type=Path, default=None)
    parser.add_argument("--dev-ranking-file", type=Path, default=None)
    parser.add_argument("--test-ranking-file", type=Path, default=None)
    parser.add_argument("--novelty-file", type=Path, default=Path("dataset/idea_novelty_dataset.jsonl"))
    parser.add_argument("--pair-text-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/stage_joint_coverage_text_deberta"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--score-loss-weight", type=float, default=0.3)
    parser.add_argument("--prediction-mode", choices=["class", "ordinal", "blend"], default="blend")
    parser.add_argument("--ordinal-blend-weight", type=float, default=0.25)
    parser.add_argument("--selection-metric", choices=["macro_f1", "qwk", "accuracy", "neg_ordinal_mae", "score_pearson"], default="qwk")
    parser.add_argument("--include-prior-signals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-summary-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    novelty_index = load_novelty_index(args.novelty_file)
    text_lookup = build_text_lookup(novelty_index, args.pair_text_file)
    split_rows, split_source = load_ranking_splits(args, novelty_index)
    if args.max_train_samples:
        split_rows["train"] = sample_train_rows(split_rows["train"], novelty_index, args.max_train_samples, args.seed)
    if args.max_eval_samples:
        split_rows["dev"] = split_rows["dev"][: args.max_eval_samples]
        split_rows["test"] = split_rows["test"][: args.max_eval_samples]

    train_dataset = JointTextDataset(
        split_rows["train"],
        novelty_index,
        text_lookup,
        args.top_k,
        args.include_prior_signals,
        args.include_summary_features,
    )
    dev_dataset = JointTextDataset(
        split_rows["dev"],
        novelty_index,
        text_lookup,
        args.top_k,
        args.include_prior_signals,
        args.include_summary_features,
    )
    test_dataset = JointTextDataset(
        split_rows["test"],
        novelty_index,
        text_lookup,
        args.top_k,
        args.include_prior_signals,
        args.include_summary_features,
    )

    device = device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    cfg = JointTextConfig(
        model_name=args.model_name,
        dropout=args.dropout,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model = JointTextModel(cfg).to(device)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    train_loader = build_loader(train_dataset, tokenizer, args.batch_size, args.max_length, True, args.num_workers)
    dev_loader = build_loader(dev_dataset, tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers)
    test_loader = build_loader(test_dataset, tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers)

    weights = class_weights(train_dataset, device)
    ordinal_weights = ordinal_pos_weights(train_dataset, device)
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
    split_sizes = {split: len(dataset) for split, dataset in [("train", train_dataset), ("dev", dev_dataset), ("test", test_dataset)]}
    run_config = {
        "args": {key: [str(item) for item in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": asdict(cfg),
        "labels": JOINT_LABELS,
        "split_source": split_source,
        "split_sizes": split_sizes,
        "train_class_distribution": dict(Counter(JOINT_LABELS[int(row["label"])] for row in train_dataset.items)),
        "class_weights": [round(float(x), 4) for x in weights.detach().cpu().tolist()],
        "ordinal_pos_weights": [round(float(x), 4) for x in ordinal_weights.detach().cpu().tolist()],
        "missing_prior_texts": {
            "train": train_dataset.missing_prior_texts,
            "dev": dev_dataset.missing_prior_texts,
            "test": test_dataset.missing_prior_texts,
        },
        "leakage_control": [
            "joint_coverage_label and joint_coverage_score are used only as supervision",
            "the prompt includes target text, prior text, metadata, and pair-level signals",
            "for production staged evaluation, use prior rankings built from pair predictions and provide a pair text file only for text lookup",
        ],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, list[float]] = defaultdict(list)
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            inputs = move_to_device(batch["inputs"], device)
            with amp_context(device, args.amp, args.bf16):
                outputs = model(inputs)
                loss, parts = compute_loss(
                    outputs,
                    batch,
                    weights,
                    ordinal_weights,
                    args.ce_loss_weight,
                    args.ordinal_loss_weight,
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

        dev_metrics = evaluate(model, dev_loader, device, weights, ordinal_weights, args)
        dev_metrics["epoch"] = epoch
        dev_metrics["global_step"] = global_step
        dev_metrics["train_loss"] = {name: round(float(np.mean(values)), 6) for name, values in running.items()}
        print(json.dumps({"split": "dev", **dev_metrics}, ensure_ascii=False, indent=2))
        (args.output_dir / f"dev_epoch_{epoch}.json").write_text(json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        score = metric_value(dev_metrics, args.selection_metric)
        if score > best_score:
            best_score = score
            best_metrics = dev_metrics
            best_dir = args.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "labels": JOINT_LABELS,
                    "metrics": dev_metrics,
                },
                best_dir / "checkpoint.pt",
            )
            tokenizer.save_pretrained(best_dir / "tokenizer")
            (best_dir / "metrics.json").write_text(json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint = torch.load(args.output_dir / "best" / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics: dict[str, Any] = {"best_dev": best_metrics}
    for split, loader in [("train", train_loader), ("dev", dev_loader), ("test", test_loader)]:
        metrics = evaluate(model, loader, device, weights, ordinal_weights, args, return_predictions=True)
        pred_rows = metrics.pop("_predictions")
        final_metrics[split] = metrics
        write_jsonl(args.output_dir / f"{split}_predictions.jsonl", pred_rows)
        print(json.dumps({"split": split, **metrics}, ensure_ascii=False, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
