"""EGRD model and joint training of paper Stages B and C."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer


try:
    from .features import (
        PAIR_LABELS,
        JOINT_LABELS,
        NOVELTY_LABELS,
        DEFAULT_RESIDUAL_TYPES,
        COARSE_TYPES,
        INNOVATION_FAMILIES,
        FINE_TYPE_GROUPS,
        TYPE_TO_COARSE,
        PAIR_TO_ID,
        JOINT_TO_ID,
        NOVELTY_TO_ID,
        COARSE_TO_ID,
        V22_REINIT_PREFIXES,
        V24_SEMANTIC_REINIT_PREFIXES,
        PAIR_FEATURE_NAMES,
        JOINT_GLOBAL_FEATURES,
        PAIR_GLOBAL_FEATURES,
        ABLATION_TARGETS,
        GLOBAL_FEATURE_NAMES,
    )
except ImportError:
    from features import (
        PAIR_LABELS,
        JOINT_LABELS,
        NOVELTY_LABELS,
        DEFAULT_RESIDUAL_TYPES,
        COARSE_TYPES,
        INNOVATION_FAMILIES,
        FINE_TYPE_GROUPS,
        TYPE_TO_COARSE,
        PAIR_TO_ID,
        JOINT_TO_ID,
        NOVELTY_TO_ID,
        COARSE_TO_ID,
        V22_REINIT_PREFIXES,
        V24_SEMANTIC_REINIT_PREFIXES,
        PAIR_FEATURE_NAMES,
        JOINT_GLOBAL_FEATURES,
        PAIR_GLOBAL_FEATURES,
        ABLATION_TARGETS,
        GLOBAL_FEATURE_NAMES,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def temporal_split(rows: list[dict[str, Any]], novelty_index: dict[str, dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    dated = []
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


def load_novelty_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("target_idea_id") or ""): row
        for row in read_jsonl(path)
        if row.get("target_idea_id")
    }


def load_joint_predictions(pred_dir: Path | None) -> dict[str, dict[str, Any]]:
    if pred_dir is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name in [
        "train_predictions.jsonl",
        "dev_predictions.jsonl",
        "test_predictions.jsonl",
        "all_predictions.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "all.jsonl",
    ]:
        path = pred_dir / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            target_id = str(row.get("target_idea_id") or "")
            if target_id:
                out[target_id] = row
    return out


def build_text_lookup(novelty_index: dict[str, dict[str, Any]], pair_text_files: list[Path]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for target_id, row in novelty_index.items():
        prior_ids = list(row.get("top_priors") or row.get("pre_recomputed_top_priors") or [])
        prior_texts = list(row.get("top_prior_texts") or [])
        for idx, prior_id_raw in enumerate(prior_ids):
            if idx < len(prior_texts) and prior_texts[idx]:
                lookup.setdefault((target_id, str(prior_id_raw)), str(prior_texts[idx]))
    for path in pair_text_files:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            target_id = str(row.get("target_idea_id") or "")
            prior_id = str(row.get("prior_idea_id") or "")
            text = str(row.get("prior_text") or "")
            if target_id and prior_id and text:
                lookup[(target_id, prior_id)] = text
    return lookup


def ranking_priors(row: dict[str, Any]) -> list[dict[str, Any]]:
    priors = row.get("ranked_priors")
    if isinstance(priors, list):
        return [dict(item) for item in priors]
    priors = row.get("candidate_priors")
    if isinstance(priors, list):
        return [dict(item) for item in priors]
    return []


def construct_ranking_from_novelty(row: dict[str, Any]) -> dict[str, Any]:
    prior_ids = list(row.get("top_priors") or row.get("pre_recomputed_top_priors") or [])
    prior_texts = list(row.get("top_prior_texts") or [])
    prior_labels = list(row.get("top_prior_coverage_labels") or [])
    prior_scores = list(row.get("top_prior_coverage_scores") or [])
    return {
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
            for idx, prior_id in enumerate(prior_ids)
        ],
    }


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
    rows = read_jsonl(args.ranking_file) if args.ranking_file else [
        construct_ranking_from_novelty(row) for row in novelty_index.values()
    ]
    rows = [row for row in rows if str(row.get("target_idea_id") or "") in novelty_index]
    if rows and all(str(row.get("split") or "") in {"train", "dev", "test"} for row in rows):
        return (
            {
                "train": [row for row in rows if row.get("split") == "train"],
                "dev": [row for row in rows if row.get("split") == "dev"],
                "test": [row for row in rows if row.get("split") == "test"],
            },
            "split_field",
        )
    train_idx, dev_idx, test_idx = temporal_split(rows, novelty_index)
    return (
        {
            "train": [rows[idx] for idx in train_idx],
            "dev": [rows[idx] for idx in dev_idx],
            "test": [rows[idx] for idx in test_idx],
        },
        "temporal_split",
    )


def pair_label(prior: dict[str, Any]) -> str:
    label = str(
        prior.get("calibrated_pair_label")
        or prior.get("pred_pair_label")
        or prior.get("raw_pred_pair_label")
        or prior.get("coverage_label")
        or prior.get("gold_pair_label")
        or "not_covering"
    )
    if label == "full_cover":
        return "large_cover"
    return label if label in PAIR_TO_ID else "not_covering"


def pair_score(prior: dict[str, Any]) -> float:
    return clamp01(
        safe_float(
            prior.get("calibrated_score"),
            safe_float(
                prior.get("pred_coverage_score"),
                safe_float(
                    prior.get("coverage_score"),
                    safe_float(prior.get("gold_coverage_score"), safe_float(prior.get("rank_score"), 0.0)),
                ),
            ),
        )
    )


def pair_probabilities(prior: dict[str, Any]) -> list[float]:
    raw = prior.get("pred_pair_probabilities") or prior.get("pair_probabilities") or {}
    values = [safe_float(raw.get(label), 0.0) for label in PAIR_LABELS]
    total = sum(values)
    if total > 0:
        return [float(value / total) for value in values]
    label = pair_label(prior)
    score = pair_score(prior)
    probs = [0.0] * len(PAIR_LABELS)
    probs[PAIR_TO_ID[label]] = 0.7
    if label == "not_covering":
        probs[1] = min(0.25, score)
    elif label == "related_not_covering":
        probs[0] = 0.15
        probs[2] = min(0.15, score * 0.25)
    elif label == "partial_cover":
        probs[1] = 0.15
        probs[3] = min(0.15, score * 0.25)
    else:
        probs[2] = 0.2
        probs[1] = 0.1
    total = sum(probs)
    return [float(value / max(total, 1e-8)) for value in probs]


def joint_union_score(scores: list[float]) -> float:
    value = 1.0
    for score in scores:
        value *= 1.0 - min(0.99, max(0.0, score))
    return 1.0 - value


def probability_entropy(values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    normalized = [max(0.0, float(value) / total) for value in values]
    return float(-sum(value * math.log(max(value, 1e-8)) for value in normalized))


def joint_probabilities(joint_row: dict[str, Any] | None, fallback_score: float) -> list[float]:
    if joint_row:
        raw = joint_row.get("pred_joint_probabilities") or joint_row.get("calibrated_joint_probabilities") or {}
        values = [safe_float(raw.get(label), 0.0) for label in JOINT_LABELS]
        total = sum(values)
        if total > 0:
            return [float(value / total) for value in values]
    score = clamp01(fallback_score)
    anchors = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    sigma = 0.20
    values = [math.exp(-((score - anchor) ** 2) / (2 * sigma * sigma)) for anchor in anchors]
    total = sum(values)
    return [float(value / max(total, 1e-8)) for value in values]


def expected_joint_score(probs: list[float]) -> float:
    return float(sum(idx * probs[idx] for idx in range(len(probs))) / max(1, len(probs) - 1))


def preferred_joint_score(joint_row: dict[str, Any] | None, fallback: float) -> float:
    if joint_row:
        for key in [
            "calibrated_joint_coverage_score",
            "pred_joint_coverage_score",
            "pred_hierarchical_joint_coverage_score",
            "pred_direct_joint_coverage_score",
        ]:
            if key in joint_row:
                return clamp01(safe_float(joint_row.get(key), fallback))
    return clamp01(fallback)


def normalized_date(value: Any) -> float:
    date = int(safe_float(value, 0.0))
    if date <= 0:
        return 0.0
    year = date // 100
    month = date % 100
    return clamp01(((year - 2018) * 12 + max(0, month - 1)) / (10 * 12))


def rank_entropy(scores: list[float]) -> float:
    values = [max(0.0, value) for value in scores]
    total = sum(values)
    if total <= 0:
        return 0.0
    probs = [value / total for value in values]
    return probability_entropy(probs) / max(1e-8, math.log(len(probs)))


def build_vocab(values: list[Any]) -> dict[str, int]:
    counts = Counter(str(value or "unknown") for value in values)
    items = ["<unk>"] + [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])) if item != "<unk>"]
    return {item: idx for idx, item in enumerate(items)}


def vocab_id(vocab: dict[str, int], value: Any) -> int:
    return vocab.get(str(value or "unknown"), 0)


def residual_type_vector(value: Any, residual_to_id: dict[str, int]) -> np.ndarray:
    out = np.zeros(len(residual_to_id), dtype=np.float32)
    values = value if isinstance(value, list) else [value]
    for raw in values:
        label = str(raw or "")
        if label in residual_to_id:
            out[residual_to_id[label]] = 1.0
    return out


def prompt_with_spans(
    target_text: str,
    target_metadata: str,
    prior_text: str,
    prior_metadata: str,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    prefix = (
        "Identify which semantic components of the target scientific idea are supported by "
        "the historical prior and which components remain uncovered.\n\n"
        "[TARGET METADATA]\n"
        f"{target_metadata}\n\n"
        "[TARGET IDEA]\n"
    )
    target_start = len(prefix)
    middle = (
        str(target_text or "")
        + "\n\n"
        + "[PRIOR METADATA]\n"
        + f"{prior_metadata}\n\n"
        + "[HISTORICAL PRIOR]\n"
    )
    prior_start = len(prefix) + len(middle)
    prompt = prefix + middle + str(prior_text or "")
    return prompt, (target_start, target_start + len(str(target_text or ""))), (prior_start, prior_start + len(str(prior_text or "")))


@dataclass
class EGRDConfig:
    model_name: str
    top_k: int
    max_length: int
    num_semantic_units: int
    pair_feature_dim: int = len(PAIR_FEATURE_NAMES)
    global_feature_dim: int = len(GLOBAL_FEATURE_NAMES)
    residual_types: list[str] = field(default_factory=list)
    coarse_types: list[str] = field(default_factory=lambda: list(COARSE_TYPES))
    fine_type_groups: dict[str, list[str]] = field(default_factory=dict)
    slot_competition: bool = True
    unit_interaction: bool = True
    ablate_features: tuple = ()
    residual_direction: str = "residual"
    semantic_assignment_mode: str = "sparsemax"
    semantic_score_temperature: float = 0.15
    semantic_score_smoothing_kernel: int = 3
    semantic_sparsemax_weight: float = 0.95
    semantic_balance_iterations: int = 3
    semantic_min_slot_mass: float = 0.25
    semantic_selection_weight: float = 0.12
    coverage_feature_noise_std: float = 0.0
    coverage_feature_dropout: float = 0.0
    flat_type_blend_weight: float = 0.15
    novelty_fusion_mode: str = "fixed"
    novelty_direct_weight: float = 0.50
    novelty_ordinal_weight: float = 0.35
    novelty_residual_weight: float = 0.15
    aspect_vocab: dict[str, int] = field(default_factory=dict)
    contribution_vocab: dict[str, int] = field(default_factory=dict)
    task_vocab: dict[str, int] = field(default_factory=dict)


def coarse_type_for(label: str) -> str:
    return TYPE_TO_COARSE.get(label, "minor_change")


NOVELTY_SURFACE_LAMBDA = 0.45
NOVELTY_FIXED_CUTS = (0.10, 0.24, 0.42)
NOVELTY_EVIDENCE_GATE = 0.72


def gold_novelty_score(novelty: dict[str, Any], *, surface_penalty_lambda: float) -> float:
    joint = clamp01(safe_float(novelty.get("joint_coverage_score"), 0.0))
    delta = clamp01(safe_float(novelty.get("substantive_delta_score"), 0.0))
    evidence = clamp01(
        safe_float(novelty.get("evidence_sufficiency"), safe_float(novelty.get("evidence_confidence"), 0.0))
    )
    surface = clamp01(safe_float(novelty.get("surface_change_score"), 0.0))
    return clamp01((1.0 - joint) * delta * evidence * (1.0 - surface_penalty_lambda * surface))


def novelty_labels_from_score(
    scores: np.ndarray,
    evidence: np.ndarray | None,
    cuts: Sequence[float],
    *,
    evidence_gate: float | None = NOVELTY_EVIDENCE_GATE,
) -> np.ndarray:
    labels = np.searchsorted(np.asarray(cuts, dtype=np.float64), scores, side="right")
    if evidence is not None and evidence_gate is not None:
        high = NOVELTY_TO_ID["high_novelty"]
        moderate = NOVELTY_TO_ID["moderate_novelty"]
        labels = np.where((labels == high) & (evidence < evidence_gate), moderate, labels)
    return labels.astype(np.int64)


def novelty_ordinal_targets(label_id: int) -> np.ndarray:
    return np.asarray(
        [
            1.0 if label_id >= NOVELTY_TO_ID["weak_novelty"] else 0.0,
            1.0 if label_id >= NOVELTY_TO_ID["moderate_novelty"] else 0.0,
            1.0 if label_id >= NOVELTY_TO_ID["high_novelty"] else 0.0,
        ],
        dtype=np.float32,
    )


class ResidualEGRDDataset(Dataset):
    def __init__(
        self,
        rankings: list[dict[str, Any]],
        novelty_index: dict[str, dict[str, Any]],
        joint_predictions: dict[str, dict[str, Any]],
        text_lookup: dict[tuple[str, str], str],
        config: EGRDConfig,
        aspect_vocab: dict[str, int],
        contribution_vocab: dict[str, int],
        task_vocab: dict[str, int],
        residual_to_id: dict[str, int],
        *,
        use_gold_joint_evidence: bool,
        facet_ablation: str = "none",
    ) -> None:
        self.items = []
        self.config = config
        self.residual_to_id = residual_to_id
        self.facet_ablation = facet_ablation
        for ranking in rankings:
            target_id = str(ranking.get("target_idea_id") or "")
            novelty = novelty_index.get(target_id)
            if not novelty:
                continue
            residual_type = novelty.get("residual_type")
            if isinstance(residual_type, str) and residual_type not in residual_to_id:
                continue
            residual_label = str(residual_type or "")
            novelty_label = str(novelty.get("novelty_label") or "")
            if novelty_label not in NOVELTY_TO_ID:
                continue
            coarse_label = coarse_type_for(residual_label)
            fine_labels = config.fine_type_groups.get(coarse_label, [])
            fine_type_id = fine_labels.index(residual_label) if residual_label in fine_labels else 0
            novelty_label_id = NOVELTY_TO_ID[novelty_label]
            is_substantive_type = 0.0 if coarse_label == "minor_change" else 1.0
            innovation_family_id = INNOVATION_FAMILIES.index(coarse_label) if coarse_label in INNOVATION_FAMILIES else 0

            target_text = str(ranking.get("target_text") or novelty.get("target_text") or "")
            target_date = ranking.get("target_date") or novelty.get("target_date")
            target_metadata = (
                f"aspect={novelty.get('primary_aspect') or 'unknown'}; "
                f"contribution={novelty.get('contribution_type') or 'unknown'}; "
                f"task={novelty.get('task_family') or 'unknown'}; "
                f"date={target_date or 'unknown'}"
            )
            priors = ranking_priors(ranking)[: config.top_k]
            pair_scores = [pair_score(prior) for prior in priors]
            pair_union = joint_union_score(pair_scores)
            joint_row = joint_predictions.get(target_id)
            joint_score = preferred_joint_score(joint_row, pair_union)
            if use_gold_joint_evidence:
                joint_score = clamp01(safe_float(novelty.get("joint_coverage_score"), joint_score))
            joint_probs = joint_probabilities(joint_row, joint_score)
            joint_expected = expected_joint_score(joint_probs)
            joint_entropy = probability_entropy(joint_probs)
            joint_norm_entropy = joint_entropy / math.log(len(JOINT_LABELS))

            prompts: list[str] = []
            target_spans: list[tuple[int, int]] = []
            prior_spans: list[tuple[int, int]] = []
            pair_features: list[list[float]] = []
            prior_meta: list[dict[str, Any]] = []
            labels = [pair_label(prior) for prior in priors]
            rank_scores = [safe_float(prior.get("rank_score"), pair_scores[idx]) for idx, prior in enumerate(priors)]

            for idx in range(config.top_k):
                if idx < len(priors):
                    prior = priors[idx]
                    prior_id = str(prior.get("prior_idea_id") or "")
                    prior_text = str(prior.get("prior_text") or text_lookup.get((target_id, prior_id), ""))
                    label = labels[idx]
                    score = pair_scores[idx]
                    rank_score = clamp01(rank_scores[idx])
                    probs = pair_probabilities(prior)
                    covering_probability = probs[2] + probs[3]
                    large_given_covering = probs[3] / max(1e-8, covering_probability)
                    prior_metadata = (
                        f"rank={idx + 1}; "
                        f"date={prior.get('prior_date') or 'unknown'}; "
                        f"aspect={prior.get('prior_primary_aspect') or prior.get('primary_aspect') or 'unknown'}; "
                        f"contribution={prior.get('prior_contribution_type') or prior.get('contribution_type') or 'unknown'}; "
                        f"task={prior.get('prior_task_family') or prior.get('task_family') or 'unknown'}"
                    )
                    if "pair_meta" not in config.ablate_features:
                        prior_metadata += (
                            f"; pred_pair_label={label}; pred_pair_score={score:.4f}; "
                            f"rank_score={rank_score:.4f}"
                        )
                    prompt, target_span, prior_span = prompt_with_spans(target_text, target_metadata, prior_text, prior_metadata)
                    feature_row = [
                        score,
                        rank_score,
                        probs[0],
                        probs[1],
                        probs[2],
                        probs[3],
                        probs[1] + probs[2] + probs[3],
                        covering_probability,
                        large_given_covering,
                        (idx + 1) / max(1, config.top_k),
                        1.0 if prior_text else 0.0,
                    ]
                    prior_meta.append(
                        {
                            "rank": idx + 1,
                            "prior_idea_id": prior_id,
                            "prior_text": prior_text,
                            "pred_pair_label": label,
                            "pred_pair_score": round(float(score), 6),
                            "rank_score": round(float(rank_score), 6),
                        }
                    )
                else:
                    prior_text = ""
                    prior_metadata = (
                        f"rank={idx + 1}; date=missing; aspect=missing; contribution=missing; "
                        "task=missing; pred_pair_label=missing; pred_pair_score=0.0000; rank_score=0.0000"
                    )
                    prompt, target_span, prior_span = prompt_with_spans(target_text, target_metadata, prior_text, prior_metadata)
                    feature_row = [0.0] * len(PAIR_FEATURE_NAMES)
                    prior_meta.append(
                        {
                            "rank": idx + 1,
                            "prior_idea_id": "",
                            "prior_text": "",
                            "pred_pair_label": "missing",
                            "pred_pair_score": 0.0,
                            "rank_score": 0.0,
                        }
                    )
                prompts.append(prompt)
                target_spans.append(target_span)
                prior_spans.append(prior_span)
                pair_features.append(feature_row)

            padded_scores = pair_scores + [0.0] * max(0, config.top_k - len(pair_scores))
            valid_pairs = len(priors)
            covering_count = sum(1 for label in labels if label in {"partial_cover", "large_cover"})
            related_count = sum(1 for label in labels if label != "not_covering")
            top_margin = (rank_scores[0] - rank_scores[1]) if len(rank_scores) > 1 else (rank_scores[0] if rank_scores else 0.0)
            global_features = [
                normalized_date(target_date),
                valid_pairs / max(1, config.top_k),
                max(pair_scores) if pair_scores else 0.0,
                float(np.mean(pair_scores)) if pair_scores else 0.0,
                float(np.std(pair_scores)) if pair_scores else 0.0,
                sum(pair_scores),
                pair_union,
                covering_count / max(1, valid_pairs),
                related_count / max(1, valid_pairs),
                clamp01(top_margin),
                rank_entropy(padded_scores[: config.top_k]),
                joint_score,
                joint_expected,
                joint_entropy,
                joint_norm_entropy,
                1.0 if joint_row else 0.0,
                joint_score - pair_union,
                *joint_probs,
            ]

            self.items.append(
                {
                    "target_idea_id": target_id,
                    "target_text": target_text,
                    "prompts": prompts,
                    "target_spans": target_spans,
                    "prior_spans": prior_spans,
                    "pair_features": np.asarray(pair_features, dtype=np.float32),
                    "global_features": np.asarray(global_features, dtype=np.float32),
                    "joint_evidence_score": float(joint_score),
                    "joint_normalized_entropy": float(joint_norm_entropy),
                    "pair_union_score": float(pair_union),
                    "prior_meta": prior_meta,
                    "aspect_id": 0 if facet_ablation == "unknown" else vocab_id(aspect_vocab, novelty.get("primary_aspect")),
                    "contribution_id": 0 if facet_ablation == "unknown" else vocab_id(contribution_vocab, novelty.get("contribution_type")),
                    "task_id": 0 if facet_ablation == "unknown" else vocab_id(task_vocab, novelty.get("task_family")),
                    "sample_weight": min(1.0, max(0.2, safe_float(novelty.get("label_confidence"), 1.0))),
                    "substantive_delta_score": clamp01(safe_float(novelty.get("substantive_delta_score"), 0.0)),
                    "surface_change_score": clamp01(safe_float(novelty.get("surface_change_score"), 0.0)),
                    "evidence_sufficiency": clamp01(
                        safe_float(novelty.get("evidence_sufficiency"), safe_float(novelty.get("evidence_confidence"), 0.0))
                    ),
                    "residual_type_vector": residual_type_vector(residual_label, residual_to_id),
                    "residual_type_id": residual_to_id[residual_label],
                    "residual_type": residual_label,
                    "coarse_type_id": COARSE_TO_ID[coarse_label],
                    "is_substantive_type": is_substantive_type,
                    "innovation_family_id": innovation_family_id,
                    "fine_type_id": fine_type_id,
                    "is_nontrivial_combination": 1.0 if residual_label == "nontrivial_combination" else 0.0,
                    "novelty_label_id": novelty_label_id,
                    "novelty_ordinal_targets": novelty_ordinal_targets(novelty_label_id),
                    "moderate_high_mask": 1.0 if novelty_label_id >= NOVELTY_TO_ID["moderate_novelty"] else 0.0,
                    "moderate_high_target": 1.0 if novelty_label_id == NOVELTY_TO_ID["high_novelty"] else 0.0,
                    "novelty_label": novelty_label,
                    "gold_joint_coverage_score": clamp01(safe_float(novelty.get("joint_coverage_score"), 0.0)),
                    "gold_novelty_score": gold_novelty_score(novelty, surface_penalty_lambda=NOVELTY_SURFACE_LAMBDA),
                }
            )
        if not self.items:
            raise ValueError("No residual examples after joining ranking, novelty, and joint files")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class EGRDCollator:
    def __init__(self, tokenizer: Any, max_length: int, top_k: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k = top_k
        self.is_fast = bool(getattr(tokenizer, "is_fast", False))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        flat_prompts: list[str] = []
        flat_target_spans: list[tuple[int, int]] = []
        flat_prior_spans: list[tuple[int, int]] = []
        for item in batch:
            flat_prompts.extend(item["prompts"])
            flat_target_spans.extend(item["target_spans"])
            flat_prior_spans.extend(item["prior_spans"])

        encoded = self.tokenizer(
            flat_prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=self.is_fast,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping", None)
        input_ids = encoded["input_ids"].view(len(batch), self.top_k, self.max_length)
        attention_mask = encoded["attention_mask"].view(len(batch), self.top_k, self.max_length)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(len(batch), self.top_k, self.max_length)

        target_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        prior_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if offsets is None:
            target_token_mask = attention_mask.bool()
            prior_token_mask = attention_mask.bool()
        else:
            offsets = offsets.view(len(batch), self.top_k, self.max_length, 2)
            for batch_idx in range(len(batch)):
                for prior_idx in range(self.top_k):
                    target_start, target_end = flat_target_spans[batch_idx * self.top_k + prior_idx]
                    prior_start, prior_end = flat_prior_spans[batch_idx * self.top_k + prior_idx]
                    starts = offsets[batch_idx, prior_idx, :, 0]
                    ends = offsets[batch_idx, prior_idx, :, 1]
                    real_tokens = ends > starts
                    target_token_mask[batch_idx, prior_idx] = (
                        real_tokens & (starts < target_end) & (ends > target_start)
                    )
                    prior_token_mask[batch_idx, prior_idx] = (
                        real_tokens & (starts < prior_end) & (ends > prior_start)
                    )

        pair_features = torch.tensor(np.stack([item["pair_features"] for item in batch]), dtype=torch.float32)
        global_features = torch.tensor(np.stack([item["global_features"] for item in batch]), dtype=torch.float32)
        pair_mask = pair_features[:, :, PAIR_FEATURE_NAMES.index("has_prior_text")] > 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "target_token_mask": target_token_mask,
            "prior_token_mask": prior_token_mask,
            "pair_mask": pair_mask,
            "pair_features": pair_features,
            "global_features": global_features,
            "aspect_id": torch.tensor([item["aspect_id"] for item in batch], dtype=torch.long),
            "contribution_id": torch.tensor([item["contribution_id"] for item in batch], dtype=torch.long),
            "task_id": torch.tensor([item["task_id"] for item in batch], dtype=torch.long),
            "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
            "substantive_delta_score": torch.tensor(
                [item["substantive_delta_score"] for item in batch], dtype=torch.float32
            ),
            "surface_change_score": torch.tensor([item["surface_change_score"] for item in batch], dtype=torch.float32),
            "evidence_sufficiency": torch.tensor([item["evidence_sufficiency"] for item in batch], dtype=torch.float32),
            "residual_type_vector": torch.tensor(
                np.stack([item["residual_type_vector"] for item in batch]), dtype=torch.float32
            ),
            "residual_type_id": torch.tensor([item["residual_type_id"] for item in batch], dtype=torch.long),
            "coarse_type_id": torch.tensor([item["coarse_type_id"] for item in batch], dtype=torch.long),
            "is_substantive_type": torch.tensor([item["is_substantive_type"] for item in batch], dtype=torch.float32),
            "innovation_family_id": torch.tensor([item["innovation_family_id"] for item in batch], dtype=torch.long),
            "fine_type_id": torch.tensor([item["fine_type_id"] for item in batch], dtype=torch.long),
            "is_nontrivial_combination": torch.tensor(
                [item["is_nontrivial_combination"] for item in batch], dtype=torch.float32
            ),
            "novelty_label_id": torch.tensor([item["novelty_label_id"] for item in batch], dtype=torch.long),
            "novelty_ordinal_targets": torch.tensor(
                np.stack([item["novelty_ordinal_targets"] for item in batch]), dtype=torch.float32
            ),
            "moderate_high_mask": torch.tensor([item["moderate_high_mask"] for item in batch], dtype=torch.float32),
            "moderate_high_target": torch.tensor([item["moderate_high_target"] for item in batch], dtype=torch.float32),
            "joint_evidence_score": torch.tensor([item["joint_evidence_score"] for item in batch], dtype=torch.float32),
            "joint_normalized_entropy": torch.tensor(
                [item["joint_normalized_entropy"] for item in batch], dtype=torch.float32
            ),
            "target_idea_id": [item["target_idea_id"] for item in batch],
            "target_text": [item["target_text"] for item in batch],
            "prior_meta": [item["prior_meta"] for item in batch],
            "gold_residual_type": [item["residual_type"] for item in batch],
            "gold_novelty_label": [item["novelty_label"] for item in batch],
            "gold_joint_coverage_score": torch.tensor(
                [item["gold_joint_coverage_score"] for item in batch], dtype=torch.float32
            ),
            "gold_novelty_score": torch.tensor(
                [item["gold_novelty_score"] for item in batch], dtype=torch.float32
            ),
            "pair_union_score": torch.tensor([item["pair_union_score"] for item in batch], dtype=torch.float32),
        }


try:
    from .stage_b_decomposition import ResidualDecompositionMixin
    from .stage_c_prediction import NoveltyPredictionMixin
except ImportError:
    from stage_b_decomposition import ResidualDecompositionMixin
    from stage_c_prediction import NoveltyPredictionMixin


class EGRDResidualPredictor(ResidualDecompositionMixin, NoveltyPredictionMixin, nn.Module):
    def __init__(
        self,
        config: EGRDConfig,
        *,
        dropout: float,
        metadata_dim: int,
        load_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name, **load_kwargs)
        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden_size
        self.num_semantic_units = config.num_semantic_units

        self.semantic_queries = nn.Parameter(torch.randn(config.num_semantic_units, hidden_size) * 0.02)
        self.pair_feature_projection = nn.Sequential(
            nn.Linear(config.pair_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.unit_alignment_projection = nn.Linear(hidden_size, hidden_size)
        self.prior_alignment_projection = nn.Linear(hidden_size, hidden_size)
        self.support_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4 + config.pair_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.coverage_pool_score = nn.Linear(hidden_size, 1)
        self.unit_active_head = nn.Linear(hidden_size, 1)

        self.aspect_embedding = nn.Embedding(len(config.aspect_vocab), metadata_dim)
        self.contribution_embedding = nn.Embedding(len(config.contribution_vocab), metadata_dim)
        self.task_embedding = nn.Embedding(len(config.task_vocab), metadata_dim)
        global_input_dim = config.global_feature_dim + metadata_dim * 3
        self.global_projection = nn.Sequential(
            nn.Linear(global_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_strength_head = nn.Linear(hidden_size, 1)
        self.interaction_pool_score = nn.Linear(hidden_size, 1)
        self.coverage_gate = nn.Sequential(
            nn.Linear(hidden_size * 3 + 1, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.residual_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_input_dim = hidden_size * 4
        self.head_norm = nn.LayerNorm(head_input_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.substantive_head = self._regression_head(head_input_dim)
        self.surface_head = self._regression_head(head_input_dim)
        self.evidence_head = self._regression_head(head_input_dim)
        self.minor_substantive_head = nn.Linear(head_input_dim + 3, 1)
        self.innovation_family_head = nn.Linear(head_input_dim, len(INNOVATION_FAMILIES))
        self.fine_type_heads = nn.ModuleDict(
            {
                coarse: nn.Linear(head_input_dim, max(1, len(labels)))
                for coarse, labels in config.fine_type_groups.items()
            }
        )
        self.flat_type_head = nn.Linear(head_input_dim, len(config.residual_types))
        self.combination_head = nn.Linear(hidden_size, 1)
        novelty_input_dim = head_input_dim + 3 + len(config.residual_types) + len(JOINT_LABELS) + 1
        self.novelty_direct_head = nn.Linear(novelty_input_dim, len(NOVELTY_LABELS))
        self.novelty_ordinal_head = nn.Linear(novelty_input_dim, 3)
        self.moderate_high_head = nn.Linear(novelty_input_dim, 1)
        residual_novelty_hidden = max(64, hidden_size // 2)
        self.residual_novelty_head = nn.Sequential(
            nn.Linear(novelty_input_dim, residual_novelty_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_novelty_hidden, len(NOVELTY_LABELS)),
        )
        self.novelty_fusion_gate = nn.Linear(novelty_input_dim, 3)
        novelty_score_hidden = max(64, hidden_size // 2)
        self.novelty_score_head = nn.Sequential(
            nn.Linear(novelty_input_dim, novelty_score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(novelty_score_hidden, 1),
        )

    @staticmethod
    def _regression_head(input_dim: int) -> nn.Sequential:
        hidden = max(64, input_dim // 2)
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def hierarchical_type_probabilities(
        self,
        minor_substantive_logits: torch.Tensor,
        innovation_family_logits: torch.Tensor,
        fine_logits: dict[str, torch.Tensor],
        combination_probability: torch.Tensor,
    ) -> torch.Tensor:
        substantive_prob = torch.sigmoid(minor_substantive_logits.float()).squeeze(-1)
        minor_prob = 1.0 - substantive_prob
        family_probs = torch.softmax(innovation_family_logits.float(), dim=-1)
        coarse_probs = torch.cat(
            [
                minor_prob.unsqueeze(-1),
                substantive_prob.unsqueeze(-1) * family_probs,
            ],
            dim=-1,
        )
        out = coarse_probs.new_zeros((coarse_probs.shape[0], len(self.config.residual_types)))
        for type_idx, label in enumerate(self.config.residual_types):
            coarse = coarse_type_for(label)
            coarse_idx = self.config.coarse_types.index(coarse)
            labels = self.config.fine_type_groups.get(coarse, [])
            if len(labels) <= 1:
                out[:, type_idx] = coarse_probs[:, coarse_idx]
                continue
            fine_probs = torch.softmax(fine_logits[coarse].float(), dim=-1)
            fine_idx = labels.index(label) if label in labels else 0
            out[:, type_idx] = coarse_probs[:, coarse_idx] * fine_probs[:, fine_idx]
        if "nontrivial_combination" in self.config.residual_types:
            combo_idx = self.config.residual_types.index("nontrivial_combination")
            combo_family = "combination" if "combination" in self.config.coarse_types else "mechanism_or_system"
            combo_family_idx = self.config.coarse_types.index(combo_family)
            combo_aux = coarse_probs[:, combo_family_idx] * combination_probability
            out[:, combo_idx] = 0.70 * out[:, combo_idx] + 0.30 * combo_aux
        return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def ordinal_probabilities(ordinal_logits: torch.Tensor, moderate_high_probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(ordinal_logits.float())
        q1 = gates[:, 0]
        q2 = q1 * gates[:, 1]
        q3 = q2 * gates[:, 2]
        ordinal_probs = torch.stack(
            [
                1.0 - q1,
                q1 - q2,
                q2 - q3,
                q3,
            ],
            dim=-1,
        ).clamp_min(0.0)
        conditional_probs = torch.stack(
            [
                1.0 - q1,
                q1 - q2,
                q2 * (1.0 - moderate_high_probability),
                q2 * moderate_high_probability,
            ],
            dim=-1,
        ).clamp_min(0.0)
        ordinal_probs = ordinal_probs / ordinal_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        conditional_probs = conditional_probs / conditional_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return ordinal_probs, conditional_probs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_features: torch.Tensor,
        global_features: torch.Tensor,
        aspect_id: torch.Tensor,
        contribution_id: torch.Tensor,
        task_id: torch.Tensor,
        joint_normalized_entropy: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Stage A predictions arrive as fixed features. B and C share gradients.
        decomposition = self.decompose(
            input_ids, attention_mask, target_token_mask, pair_mask,
            pair_features, token_type_ids,
        )
        return self.predict_novelty(
            decomposition,
            pair_features=pair_features,
            global_features=global_features,
            aspect_id=aspect_id,
            contribution_id=contribution_id,
            task_id=task_id,
            joint_normalized_entropy=joint_normalized_entropy,
        )


def freeze_encoder(model: EGRDResidualPredictor, trainable_last_layers: int) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    if trainable_last_layers <= 0:
        return
    layers = None
    for attr_path in [
        "encoder.layer",
        "deberta.encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
    ]:
        module: Any = model.encoder
        ok = True
        for part in attr_path.split("."):
            if not hasattr(module, part):
                ok = False
                break
            module = getattr(module, part)
        if ok:
            layers = module
            break
    if layers is None:
        return
    for layer in list(layers)[-trainable_last_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


def unfreeze_encoder(model: EGRDResidualPredictor, trainable_last_layers: int) -> None:
    if trainable_last_layers < 0:
        for param in model.encoder.parameters():
            param.requires_grad = True
    else:
        freeze_encoder(model, trainable_last_layers)


def set_regression_heads_trainable(model: EGRDResidualPredictor, trainable: bool) -> None:
    for module in [model.substantive_head, model.surface_head, model.evidence_head]:
        for param in module.parameters():
            param.requires_grad = trainable


def load_encoder_checkpoint(model: EGRDResidualPredictor, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value
        elif key.startswith("backbone."):
            encoder_state[key[len("backbone."):]] = value
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_tensors": len(encoder_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def load_matching_checkpoint(
    model: EGRDResidualPredictor,
    checkpoint_path: Path,
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    current = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped_by_prefix: list[str] = []
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []
    for key, value in state.items():
        if any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
            continue
        if key not in current:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            skipped_shape.append(key)
            continue
        loadable[key] = value
    current.update(loadable)
    model.load_state_dict(current)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_tensors": len(loadable),
        "skip_prefixes": list(skip_prefixes),
        "skipped_by_prefix": len(skipped_by_prefix),
        "skipped_missing": len(skipped_missing),
        "skipped_shape": len(skipped_shape),
        "skipped_examples": (skipped_by_prefix + skipped_missing + skipped_shape)[:50],
    }


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (loss * weights.to(dtype=loss.dtype, device=loss.device)).sum() / weights.sum().clamp_min(1e-6)


def zero_loss_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


def bounded_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().clamp(-20.0, 20.0)


def class_balanced_weights(labels: list[int], n_classes: int, *, min_weight: float = 0.25, max_weight: float = 4.0) -> list[float]:
    counts = Counter(int(label) for label in labels)
    total = max(1, len(labels))
    weights = []
    for idx in range(n_classes):
        value = math.sqrt(total / max(1.0, n_classes * counts.get(idx, 0)))
        weights.append(float(min(max_weight, max(min_weight, value))))
    return weights


def ordinal_pos_weights(label_ids: list[int]) -> list[float]:
    values = np.asarray(label_ids, dtype=np.int64)
    out = []
    for threshold in [1, 2, 3]:
        pos = int((values >= threshold).sum())
        neg = int((values < threshold).sum())
        out.append(float(min(4.0, max(0.25, math.sqrt(neg / max(1, pos))))))
    return out


def build_loss_context(dataset: ResidualEGRDDataset, config: EGRDConfig) -> dict[str, Any]:
    items = dataset.items
    context: dict[str, Any] = {
        "coarse_weights": class_balanced_weights([int(item["coarse_type_id"]) for item in items], len(config.coarse_types)),
        "innovation_family_weights": class_balanced_weights(
            [int(item["innovation_family_id"]) for item in items if float(item["is_substantive_type"]) > 0.5],
            len(INNOVATION_FAMILIES),
        ),
        "flat_type_weights": class_balanced_weights([int(item["residual_type_id"]) for item in items], len(config.residual_types)),
        "novelty_direct_weights": class_balanced_weights([int(item["novelty_label_id"]) for item in items], len(NOVELTY_LABELS)),
        "ordinal_pos_weights": ordinal_pos_weights([int(item["novelty_label_id"]) for item in items]),
    }
    substantive_count = sum(1 for item in items if float(item["is_substantive_type"]) > 0.5)
    minor_count = max(0, len(items) - substantive_count)
    minor_weights = torch.tensor(
        [
            math.sqrt(len(items) / max(1.0, 2.0 * minor_count)),
            math.sqrt(len(items) / max(1.0, 2.0 * substantive_count)),
        ],
        dtype=torch.float32,
    )
    minor_weights = (minor_weights / minor_weights.mean().clamp_min(1e-8)).clamp(max=3.0)
    context["minor_substantive_class_weights"] = [float(value) for value in minor_weights.tolist()]
    fine_weights: dict[str, list[float]] = {}
    for coarse, labels in config.fine_type_groups.items():
        coarse_id = COARSE_TO_ID[coarse]
        fine_ids = [int(item["fine_type_id"]) for item in items if int(item["coarse_type_id"]) == coarse_id]
        fine_weights[coarse] = class_balanced_weights(fine_ids, max(1, len(labels)))
    context["fine_weights"] = fine_weights
    combo_pos = sum(1 for item in items if float(item["is_nontrivial_combination"]) > 0.5)
    combo_neg = max(0, len(items) - combo_pos)
    context["combination_pos_weight"] = float(min(5.0, max(0.5, math.sqrt(combo_neg / max(1, combo_pos)))))
    context["minor_substantive_pos_weight"] = float(min(4.0, max(0.5, math.sqrt(minor_count / max(1, substantive_count)))))
    high_pos = sum(1 for item in items if int(item["novelty_label_id"]) == NOVELTY_TO_ID["high_novelty"])
    moderate_neg = sum(1 for item in items if int(item["novelty_label_id"]) == NOVELTY_TO_ID["moderate_novelty"])
    context["moderate_high_pos_weight"] = float(min(2.0, max(0.5, math.sqrt(moderate_neg / max(1, high_pos)))))
    return context


def semantic_diversity_loss(queries: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(queries, dim=-1)
    sim = normalized @ normalized.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim.masked_fill(eye, 0.0).pow(2).sum() / max(1, sim.numel() - sim.shape[0])


def attention_diversity_loss(attention: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(attention.float(), p=2, dim=-1)
    sim = torch.matmul(normalized, normalized.transpose(1, 2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).unsqueeze(0)
    orthogonality = (sim - eye).pow(2).mean()
    off_diag_mask = ~torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0).expand_as(sim)
    off_diag = sim.masked_select(off_diag_mask)
    overlap = off_diag.mean() if off_diag.numel() else zero_loss_like(sim)
    return orthogonality + 0.5 * overlap


def unit_usage_balance_loss(attention: torch.Tensor) -> torch.Tensor:
    if attention.shape[1] <= 1:
        return zero_loss_like(attention)
    usage = attention.float().sum(dim=-1)
    usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    uniform = torch.full_like(usage, 1.0 / max(1, usage.shape[-1]))
    return (
        usage
        * (
            torch.log(usage.clamp_min(1e-8))
            - torch.log(uniform.clamp_min(1e-8))
        )
    ).sum(dim=-1).mean()


def slot_assignment_entropy_loss(
    assignment: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    if assignment.shape[1] <= 1:
        return zero_loss_like(assignment)
    probabilities = assignment.float() / assignment.float().sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)
    entropy = entropy / math.log(max(2, assignment.shape[1]))
    valid = token_mask.to(dtype=entropy.dtype)
    return (entropy * valid).sum() / valid.sum().clamp_min(1.0)


def semantic_unit_separation_loss(
    semantic_units: torch.Tensor,
    max_pairwise_cosine: float,
) -> torch.Tensor:
    if semantic_units.shape[1] <= 1:
        return zero_loss_like(semantic_units)
    normalized = F.normalize(semantic_units.float(), dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(1, 2))
    off_diagonal = ~torch.eye(
        cosine.shape[-1],
        device=cosine.device,
        dtype=torch.bool,
    ).unsqueeze(0).expand_as(cosine)
    pairwise = cosine.masked_select(off_diagonal)
    if not pairwise.numel():
        return zero_loss_like(cosine)
    return F.relu(pairwise - float(max_pairwise_cosine)).pow(2).mean()


def unit_collapse_loss(coverage_scores: torch.Tensor, min_variance: float) -> torch.Tensor:
    variance = coverage_scores.float().var(dim=-1, unbiased=False)
    target_variance = max(1e-8, float(min_variance))
    return (F.relu(target_variance - variance) / target_variance).mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_target_delta: float = 0.05,
    margin: float = 0.02,
) -> torch.Tensor:
    if pred.numel() < 2:
        return zero_loss_like(pred)
    target_delta = target.unsqueeze(1) - target.unsqueeze(0)
    pred_delta = pred.unsqueeze(1) - pred.unsqueeze(0)
    mask = target_delta.abs() >= min_target_delta
    if not bool(mask.any()):
        return zero_loss_like(pred)
    direction = target_delta.sign()
    pair_weights = torch.sqrt(weights.unsqueeze(1) * weights.unsqueeze(0)).to(dtype=pred.dtype, device=pred.device)
    losses = F.relu(float(margin) - direction * pred_delta)
    effective = pair_weights * mask.to(dtype=pred.dtype)
    return (losses * effective).sum() / effective.sum().clamp_min(1e-6)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    model: EGRDResidualPredictor,
    args: argparse.Namespace,
    loss_context: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = batch["sample_weight"].to(outputs["substantive_score"].device)
    sub_loss = weighted_mean(
        F.smooth_l1_loss(outputs["substantive_score"], batch["substantive_delta_score"].to(weights.device), reduction="none", beta=args.huber_beta),
        weights,
    )
    surface_loss = weighted_mean(
        F.smooth_l1_loss(outputs["surface_score"], batch["surface_change_score"].to(weights.device), reduction="none", beta=args.huber_beta),
        weights,
    )
    evidence_loss = weighted_mean(
        F.smooth_l1_loss(outputs["evidence_sufficiency"], batch["evidence_sufficiency"].to(weights.device), reduction="none", beta=args.huber_beta),
        weights,
    )
    regression_rank_loss = (
        pairwise_ranking_loss(outputs["substantive_score"], batch["substantive_delta_score"].to(weights.device), weights)
        + float(args.surface_ranking_weight)
        * pairwise_ranking_loss(outputs["surface_score"], batch["surface_change_score"].to(weights.device), weights)
        + float(args.evidence_ranking_weight)
        * pairwise_ranking_loss(outputs["evidence_sufficiency"], batch["evidence_sufficiency"].to(weights.device), weights)
    )
    gold_score = batch["gold_novelty_score"].to(weights.device)
    novelty_score_reg_loss = weighted_mean(
        F.smooth_l1_loss(
            outputs["predicted_novelty_score"], gold_score, reduction="none", beta=args.huber_beta
        ),
        weights,
    )
    novelty_score_rank_loss = pairwise_ranking_loss(
        outputs["predicted_novelty_score"], gold_score, weights
    )
    novelty_score_loss = (
        novelty_score_reg_loss + float(args.novelty_score_rank_weight) * novelty_score_rank_loss
    )
    gold_coverage_loss = weighted_mean(
        F.smooth_l1_loss(
            outputs["estimated_joint_coverage"],
            batch["gold_joint_coverage_score"].to(weights.device),
            reduction="none",
            beta=args.huber_beta,
        ),
        weights,
    )
    regression_rank_loss = regression_rank_loss + 0.5 * pairwise_ranking_loss(
        outputs["estimated_joint_coverage"],
        batch["gold_joint_coverage_score"].to(weights.device),
        weights,
    )
    predicted_consistency_loss = weighted_mean(
        F.smooth_l1_loss(
            outputs["estimated_joint_coverage"],
            batch["joint_evidence_score"].to(weights.device),
            reduction="none",
            beta=args.huber_beta,
        ),
        weights * (1.0 - batch["joint_normalized_entropy"].to(weights.device).clamp(0.0, 1.0)),
    )
    coverage_loss = (
        args.gold_coverage_loss_weight * gold_coverage_loss
        + args.predicted_coverage_consistency_loss_weight * predicted_consistency_loss
    )

    substantive_target = batch["is_substantive_type"].to(weights.device)
    minor_substantive_class_weights = torch.tensor(
        loss_context["minor_substantive_class_weights"],
        device=weights.device,
        dtype=torch.float32,
    )
    minor_logits = bounded_logits(outputs["minor_substantive_logits"]).squeeze(-1)
    minor_substantive_logits = torch.stack([torch.zeros_like(minor_logits), minor_logits], dim=-1)
    minor_substantive_loss = weighted_mean(
        F.cross_entropy(
            minor_substantive_logits,
            substantive_target.long(),
            weight=minor_substantive_class_weights,
            reduction="none",
        ),
        weights,
    )
    family_mask = substantive_target > 0.5
    if bool(family_mask.any()):
        family_weights = torch.tensor(loss_context["innovation_family_weights"], device=weights.device, dtype=torch.float32)
        innovation_family_loss = weighted_mean(
            F.cross_entropy(
                bounded_logits(outputs["innovation_family_logits"])[family_mask],
                batch["innovation_family_id"].to(weights.device)[family_mask],
                weight=family_weights,
                reduction="none",
            ),
            weights[family_mask],
        )
    else:
        innovation_family_loss = zero_loss_like(outputs["innovation_family_logits"])
    fine_losses = []
    for coarse, labels in model.config.fine_type_groups.items():
        coarse_id = COARSE_TO_ID[coarse]
        mask = batch["coarse_type_id"].to(weights.device) == coarse_id
        if not bool(mask.any()):
            continue
        fine_weights = torch.tensor(loss_context["fine_weights"][coarse], device=weights.device, dtype=torch.float32)
        fine_losses.append(
            weighted_mean(
                F.cross_entropy(
                    bounded_logits(outputs["fine_type_logits"][coarse])[mask],
                    batch["fine_type_id"].to(weights.device)[mask],
                    weight=fine_weights,
                    reduction="none",
                ),
                weights[mask],
            )
        )
    fine_loss = torch.stack(fine_losses).mean() if fine_losses else zero_loss_like(outputs["coarse_type_logits"])
    flat_type_weights = torch.tensor(loss_context["flat_type_weights"], device=weights.device, dtype=torch.float32)
    flat_type_loss = weighted_mean(
        F.cross_entropy(
            bounded_logits(outputs["flat_type_logits"]),
            batch["residual_type_id"].to(weights.device),
            weight=flat_type_weights,
            reduction="none",
        ),
        weights,
    )
    combination_pos_weight = torch.tensor(loss_context["combination_pos_weight"], device=weights.device, dtype=torch.float32)
    combination_bce_loss = weighted_mean(
        F.binary_cross_entropy_with_logits(
            bounded_logits(outputs["combination_logits"]),
            batch["is_nontrivial_combination"].to(weights.device),
            pos_weight=combination_pos_weight,
            reduction="none",
        ),
        weights,
    )
    substantive_probability = torch.sigmoid(minor_logits).clamp(1e-6, 1.0 - 1e-6)
    innovation_family_probabilities = torch.softmax(bounded_logits(outputs["innovation_family_logits"]), dim=-1)
    if "combination" in INNOVATION_FAMILIES:
        combination_family_probability = innovation_family_probabilities[:, INNOVATION_FAMILIES.index("combination")]
    else:
        combination_family_probability = torch.zeros_like(substantive_probability)
    combination_probability = torch.sigmoid(bounded_logits(outputs["combination_logits"])).clamp(1e-6, 1.0 - 1e-6)
    combination_consistency_loss = weighted_mean(
        F.relu(combination_probability - substantive_probability),
        weights,
    )
    combination_family_consistency_loss = weighted_mean(
        F.mse_loss(combination_probability, combination_family_probability, reduction="none"),
        weights,
    )
    combination_loss = (
        combination_bce_loss
        + args.combination_consistency_loss_weight * combination_consistency_loss
        + args.combination_family_consistency_loss_weight * combination_family_consistency_loss
    )
    hierarchical_type_loss = (
        args.minor_substantive_loss_weight * minor_substantive_loss
        + args.innovation_family_loss_weight * innovation_family_loss
        + args.conditional_fine_loss_weight * fine_loss
        + args.flat_type_loss_weight * flat_type_loss
    )
    type_loss = hierarchical_type_loss + args.combination_binary_loss_weight * combination_loss

    ordinal_targets = batch["novelty_ordinal_targets"].to(weights.device)
    ordinal_pos_weight = torch.tensor(loss_context["ordinal_pos_weights"], device=weights.device, dtype=torch.float32).unsqueeze(0)
    ordinal_probs = torch.stack(
        [
            outputs["novelty_ordinal_probabilities"][:, 1:].sum(dim=-1),
            outputs["novelty_ordinal_probabilities"][:, 2:].sum(dim=-1),
            outputs["novelty_ordinal_probabilities"][:, 3],
        ],
        dim=-1,
    ).clamp(1e-6, 1.0 - 1e-6)
    ordinal_loss_raw = -(
        ordinal_pos_weight * ordinal_targets * torch.log(ordinal_probs)
        + (1.0 - ordinal_targets) * torch.log(1.0 - ordinal_probs)
    ).mean(dim=-1)
    ordinal_loss = weighted_mean(ordinal_loss_raw, weights)
    direct_weights = torch.tensor(loss_context["novelty_direct_weights"], device=weights.device, dtype=torch.float32)
    direct_loss = weighted_mean(
        F.cross_entropy(
            bounded_logits(outputs["novelty_direct_logits"]),
            batch["novelty_label_id"].to(weights.device),
            weight=direct_weights,
            reduction="none",
        ),
        weights,
    )
    residual_novelty_loss = weighted_mean(
        F.cross_entropy(
            bounded_logits(outputs["residual_novelty_logits"]),
            batch["novelty_label_id"].to(weights.device),
            weight=direct_weights,
            reduction="none",
        ),
        weights,
    )
    final_novelty_loss = weighted_mean(
        -torch.log(
            outputs["novelty_probabilities"]
            .gather(1, batch["novelty_label_id"].to(weights.device).unsqueeze(1))
            .squeeze(1)
            .clamp_min(1e-8)
        ),
        weights,
    )
    mh_mask = batch["moderate_high_mask"].to(weights.device) > 0.5
    if bool(mh_mask.any()):
        mh_target = batch["moderate_high_target"].to(weights.device)[mh_mask]
        mh_prob = outputs["moderate_high_probability"][mh_mask].clamp(1e-6, 1.0 - 1e-6)
        mh_weight = weights[mh_mask].clone()
        hard_negative = (mh_target < 0.5) & (mh_prob.detach() >= args.hard_negative_high_probability)
        mh_weight = torch.where(hard_negative, mh_weight * args.hard_negative_weight, mh_weight)
        mh_pos_weight = torch.tensor(loss_context["moderate_high_pos_weight"], device=weights.device, dtype=torch.float32)
        mh_bce = F.binary_cross_entropy_with_logits(
            bounded_logits(outputs["moderate_high_logits"])[mh_mask],
            mh_target,
            pos_weight=mh_pos_weight,
            reduction="none",
        )
        moderate_high_bce_loss = weighted_mean(mh_bce, mh_weight)
        pos_scores = mh_prob[mh_target > 0.5]
        neg_scores = mh_prob[mh_target < 0.5]
        if pos_scores.numel() and neg_scores.numel():
            high_margin_loss = F.relu(
                args.high_margin - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)
            ).mean()
        else:
            high_margin_loss = zero_loss_like(mh_prob)
        moderate_high_loss = moderate_high_bce_loss + args.high_margin_loss_weight * high_margin_loss
    else:
        moderate_high_bce_loss = zero_loss_like(outputs["moderate_high_probability"])
        high_margin_loss = zero_loss_like(outputs["moderate_high_probability"])
        moderate_high_loss = zero_loss_like(outputs["moderate_high_probability"])
    q1 = ordinal_probs[:, 0]
    q2 = ordinal_probs[:, 1]
    q3 = ordinal_probs[:, 2]
    monotonic_loss = (F.relu(q2 - q1) + F.relu(q3 - q2)).mean()
    gate_target = outputs["novelty_fusion_gates"].new_tensor(args.gate_balance_target)
    gate_target = gate_target / gate_target.sum().clamp_min(1e-8)
    mean_gate = outputs["novelty_fusion_gates"].float().mean(dim=0)
    gate_balance_loss = F.mse_loss(mean_gate, gate_target)
    novelty_loss = (
        args.ordinal_novelty_loss_weight * ordinal_loss
        + args.direct_novelty_loss_weight * direct_loss
        + args.residual_novelty_loss_weight * residual_novelty_loss
        + args.final_novelty_loss_weight * final_novelty_loss
        + args.moderate_high_loss_weight * moderate_high_loss
        + args.monotonic_loss_weight * monotonic_loss
        + args.gate_balance_loss_weight * gate_balance_loss
    )
    query_diversity = semantic_diversity_loss(model.semantic_queries)
    attention_diversity = attention_diversity_loss(outputs["semantic_attention"])
    usage_balance = unit_usage_balance_loss(outputs["semantic_raw_token_assignment"])
    slot_entropy = slot_assignment_entropy_loss(
        outputs["semantic_token_assignment"],
        outputs["semantic_target_mask"],
    )
    unit_separation = semantic_unit_separation_loss(
        outputs["semantic_units"],
        args.max_semantic_unit_cosine,
    )
    diversity = (
        args.attention_overlap_loss_weight * attention_diversity
        + args.query_diversity_loss_weight * query_diversity
        + args.unit_usage_balance_loss_weight * usage_balance
        + args.slot_entropy_loss_weight * slot_entropy
        + args.semantic_unit_separation_loss_weight * unit_separation
    )
    collapse_loss = unit_collapse_loss(outputs["semantic_unit_coverage_scores"], args.min_unit_coverage_variance)
    total = (
        args.substantive_loss_weight * sub_loss
        + args.surface_loss_weight * surface_loss
        + args.evidence_loss_weight * evidence_loss
        + args.regression_ranking_loss_weight * regression_rank_loss
        + args.coverage_loss_weight * coverage_loss
        + args.type_loss_weight * type_loss
        + args.novelty_loss_weight * novelty_loss
        + args.novelty_score_loss_weight * novelty_score_loss
        + args.semantic_diversity_loss_weight * diversity
        + args.unit_collapse_loss_weight * collapse_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "substantive": float(sub_loss.detach().cpu()),
        "surface": float(surface_loss.detach().cpu()),
        "evidence": float(evidence_loss.detach().cpu()),
        "regression_rank": float(regression_rank_loss.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "hierarchical_type": float(hierarchical_type_loss.detach().cpu()),
        "minor_substantive": float(minor_substantive_loss.detach().cpu()),
        "innovation_family": float(innovation_family_loss.detach().cpu()),
        "fine_type": float(fine_loss.detach().cpu()),
        "flat_type_aux": float(flat_type_loss.detach().cpu()),
        "combination_binary": float(combination_loss.detach().cpu()),
        "combination_bce": float(combination_bce_loss.detach().cpu()),
        "combination_consistency": float(combination_consistency_loss.detach().cpu()),
        "combination_family_consistency": float(combination_family_consistency_loss.detach().cpu()),
        "coverage": float(coverage_loss.detach().cpu()),
        "gold_coverage": float(gold_coverage_loss.detach().cpu()),
        "predicted_coverage_consistency": float(predicted_consistency_loss.detach().cpu()),
        "semantic_diversity": float(diversity.detach().cpu()),
        "attention_diversity": float(attention_diversity.detach().cpu()),
        "query_diversity": float(query_diversity.detach().cpu()),
        "unit_usage_balance": float(usage_balance.detach().cpu()),
        "slot_assignment_entropy": float(slot_entropy.detach().cpu()),
        "semantic_unit_separation": float(unit_separation.detach().cpu()),
        "unit_collapse": float(collapse_loss.detach().cpu()),
        "novelty": float(novelty_loss.detach().cpu()),
        "novelty_score": float(novelty_score_loss.detach().cpu()),
        "novelty_score_regression": float(novelty_score_reg_loss.detach().cpu()),
        "novelty_score_rank": float(novelty_score_rank_loss.detach().cpu()),
        "novelty_ordinal": float(ordinal_loss.detach().cpu()),
        "novelty_direct_aux": float(direct_loss.detach().cpu()),
        "novelty_residual_branch": float(residual_novelty_loss.detach().cpu()),
        "novelty_final_nll": float(final_novelty_loss.detach().cpu()),
        "moderate_high": float(moderate_high_loss.detach().cpu()),
        "moderate_high_bce": float(moderate_high_bce_loss.detach().cpu()),
        "high_margin": float(high_margin_loss.detach().cpu()),
        "monotonic": float(monotonic_loss.detach().cpu()),
        "gate_balance": float(gate_balance_loss.detach().cpu()),
        "weighted_substantive": float((args.substantive_loss_weight * sub_loss).detach().cpu()),
        "weighted_surface": float((args.surface_loss_weight * surface_loss).detach().cpu()),
        "weighted_evidence": float((args.evidence_loss_weight * evidence_loss).detach().cpu()),
        "weighted_regression_rank": float((args.regression_ranking_loss_weight * regression_rank_loss).detach().cpu()),
        "weighted_coverage": float((args.coverage_loss_weight * coverage_loss).detach().cpu()),
        "weighted_type": float((args.type_loss_weight * type_loss).detach().cpu()),
        "weighted_novelty": float((args.novelty_loss_weight * novelty_loss).detach().cpu()),
        "weighted_novelty_score": float((args.novelty_score_loss_weight * novelty_score_loss).detach().cpu()),
        "weighted_semantic_diversity": float((args.semantic_diversity_loss_weight * diversity).detach().cpu()),
        "weighted_unit_collapse": float((args.unit_collapse_loss_weight * collapse_loss).detach().cpu()),
    }
    return total, parts


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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if not len(y_true):
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}
    pred = y_pred.astype(np.float64)
    true = y_true.astype(np.float64)
    err = pred - true
    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denom = float(np.sqrt((true_centered**2).sum()) * np.sqrt((pred_centered**2).sum()))
    pearson = float((true_centered * pred_centered).sum() / denom) if denom else 0.0
    true_ranks = ranks(true)
    pred_ranks = ranks(pred)
    true_rank_centered = true_ranks - true_ranks.mean()
    pred_rank_centered = pred_ranks - pred_ranks.mean()
    rank_denom = float(np.sqrt((true_rank_centered**2).sum()) * np.sqrt((pred_rank_centered**2).sum()))
    spearman = float((true_rank_centered * pred_rank_centered).sum() / rank_denom) if rank_denom else 0.0
    return {
        "mae": round(float(np.abs(err).mean()), 4),
        "rmse": round(float(np.sqrt((err**2).mean())), 4),
        "pearson": round(float(pearson), 4),
        "spearman": round(float(spearman), 4),
    }


def qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    observed = np.zeros((n_classes, n_classes), dtype=np.float64)
    for true, pred in zip(y_true, y_pred):
        observed[int(true), int(pred)] += 1.0
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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, ordinal: bool = False) -> dict[str, Any]:
    if not len(y_true):
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "per_class": {}}
    per_class = per_class_metrics(y_true, y_pred, labels)
    f1s = [float(item["f1"]) for item in per_class.values()]
    recalls = [float(item["recall"]) for item in per_class.values()]
    tp_total = int((y_true == y_pred).sum())
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        confusion[int(true), int(pred)] += 1
    out: dict[str, Any] = {
        "accuracy": round(float(tp_total / max(1, len(y_true))), 4),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "micro_f1": round(float(tp_total / max(1, len(y_true))), 4),
        "balanced_accuracy": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            labels[idx]: {labels[jdx]: int(confusion[idx, jdx]) for jdx in range(len(labels))}
            for idx in range(len(labels))
        },
        "true_distribution": dict(Counter(labels[int(idx)] for idx in y_true)),
        "pred_distribution": dict(Counter(labels[int(idx)] for idx in y_pred)),
    }
    if ordinal:
        out["ordinal_mae"] = round(float(np.abs(y_true - y_pred).mean()), 4)
        out["qwk"] = round(qwk(y_true, y_pred, len(labels)), 4)
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, Any]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    if not len(y_true):
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_support": int(y_true.sum()),
        "negative_support": int((~y_true).sum()),
        "pred_positive": int(y_pred.sum()),
    }
    if y_score is not None and len(y_score):
        out["positive_score_mean"] = round(float(y_score[y_true].mean()), 4) if y_true.any() else 0.0
        out["negative_score_mean"] = round(float(y_score[~y_true].mean()), 4) if (~y_true).any() else 0.0
    return out


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    pos = int(y_true.sum())
    neg = int((~y_true).sum())
    if pos == 0 or neg == 0:
        return 0.0
    score_ranks = ranks(y_score.astype(np.float64))
    auc = (float(score_ranks[y_true].sum()) - pos * (pos - 1) / 2.0) / max(1.0, pos * neg)
    return float(auc)


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
    return float(precision_at_k[sorted_true].sum() / pos)


def calibrated_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold_min: float,
    threshold_max: float,
    steps: int = 81,
) -> dict[str, Any]:
    if not len(y_true) or not len(y_score):
        return {
            "fixed_threshold_0.5": binary_metrics(np.asarray([]), np.asarray([]), np.asarray([])),
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "positive_rate_baseline": 0.0,
            "best_threshold": threshold_min,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "best_pred_positive": 0,
            "score_distribution": {},
        }
    y_true_bool = y_true.astype(bool)
    fixed = binary_metrics(y_true_bool, y_score >= 0.5, y_score)
    best_threshold = threshold_min
    best_metrics = binary_metrics(y_true_bool, y_score >= threshold_min, y_score)
    best_f1 = float(best_metrics["f1"])
    for threshold in np.linspace(threshold_min, threshold_max, steps):
        current = binary_metrics(y_true_bool, y_score >= threshold, y_score)
        current_f1 = float(current["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)
            best_metrics = current
    return {
        "fixed_threshold_0.5": fixed,
        "roc_auc": round(binary_auroc(y_true_bool.astype(np.int64), y_score), 4),
        "pr_auc": round(binary_auprc(y_true_bool.astype(np.int64), y_score), 4),
        "positive_rate_baseline": round(float(y_true_bool.mean()), 4) if len(y_true_bool) else 0.0,
        "best_threshold": round(float(best_threshold), 4),
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f1": best_metrics["f1"],
        "best_pred_positive": best_metrics["pred_positive"],
        "score_distribution": {
            "positive": {
                "count": int(y_true_bool.sum()),
                "mean": round(float(y_score[y_true_bool].mean()), 4) if y_true_bool.any() else 0.0,
                "p25": round(float(np.quantile(y_score[y_true_bool], 0.25)), 4) if y_true_bool.any() else 0.0,
                "median": round(float(np.quantile(y_score[y_true_bool], 0.50)), 4) if y_true_bool.any() else 0.0,
                "p75": round(float(np.quantile(y_score[y_true_bool], 0.75)), 4) if y_true_bool.any() else 0.0,
            },
            "negative": {
                "count": int((~y_true_bool).sum()),
                "mean": round(float(y_score[~y_true_bool].mean()), 4) if (~y_true_bool).any() else 0.0,
                "p25": round(float(np.quantile(y_score[~y_true_bool], 0.25)), 4) if (~y_true_bool).any() else 0.0,
                "median": round(float(np.quantile(y_score[~y_true_bool], 0.50)), 4) if (~y_true_bool).any() else 0.0,
                "p75": round(float(np.quantile(y_score[~y_true_bool], 0.75)), 4) if (~y_true_bool).any() else 0.0,
            },
        },
    }


def score_quantiles_by_label(y_label: np.ndarray, score: np.ndarray, labels: list[str]) -> dict[str, Any]:
    out = {}
    for idx, label in enumerate(labels):
        values = score[y_label == idx]
        if len(values) == 0:
            out[label] = {"count": 0}
            continue
        q25, median, q75 = np.quantile(values.astype(np.float64), [0.25, 0.50, 0.75])
        out[label] = {
            "count": int(len(values)),
            "p25": round(float(q25), 4),
            "median": round(float(median), 4),
            "p75": round(float(q75), 4),
            "mean": round(float(values.mean()), 4),
        }
    return out


def fine_type_metrics_by_coarse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual_types: list[str],
    fine_type_groups: dict[str, list[str]],
) -> dict[str, Any]:
    out = {}
    true_labels = [residual_types[int(idx)] for idx in y_true]
    pred_labels = [residual_types[int(idx)] for idx in y_pred]
    for coarse, labels in fine_type_groups.items():
        labels_with_other = labels + ["other_coarse"]
        local_true = []
        local_pred = []
        for true_label, pred_label in zip(true_labels, pred_labels):
            if coarse_type_for(true_label) != coarse:
                continue
            local_true.append(labels.index(true_label) if true_label in labels else len(labels))
            local_pred.append(labels.index(pred_label) if pred_label in labels else len(labels))
        out[coarse] = classification_metrics(
            np.asarray(local_true, dtype=np.int64),
            np.asarray(local_pred, dtype=np.int64),
            labels_with_other,
        )
    return out


def residual_type_pred_counts(y_pred: np.ndarray, labels: list[str]) -> dict[str, int]:
    counts = Counter(labels[int(idx)] for idx in y_pred)
    return {label: int(counts.get(label, 0)) for label in labels}


def routed_residual_predictions(
    type_probabilities: np.ndarray,
    substantive_probability: np.ndarray,
    innovation_family_probabilities: np.ndarray,
    combination_probability: np.ndarray,
    residual_types: list[str],
    *,
    substantive_threshold: float,
    combination_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routed_probs = type_probabilities.copy()
    combination_type_id = residual_types.index("nontrivial_combination") if "nontrivial_combination" in residual_types else -1
    minor_type_id = residual_types.index("minor_variant") if "minor_variant" in residual_types else 0
    if combination_type_id >= 0:
        routed_probs[:, combination_type_id] = -1.0
    final_type = routed_probs.argmax(axis=1).astype(np.int64)
    minor_mask = substantive_probability < substantive_threshold
    final_type[minor_mask] = minor_type_id
    if combination_type_id >= 0 and "combination" in INNOVATION_FAMILIES:
        combination_family_probability = innovation_family_probabilities[:, INNOVATION_FAMILIES.index("combination")]
        combination_joint_score = substantive_probability * combination_family_probability * combination_probability
        combination_mask = (~minor_mask) & (combination_joint_score >= combination_threshold)
        final_type[combination_mask] = combination_type_id
        final_combination = combination_mask.astype(np.int64)
    else:
        combination_joint_score = np.zeros_like(substantive_probability)
        final_combination = np.zeros_like(final_type)
    return final_type.astype(np.int64), final_combination.astype(np.int64), combination_joint_score.astype(np.float32)


def residual_routing_objective(metrics: dict[str, Any], residual_types: list[str]) -> float:
    combo_f1 = 0.0
    if "nontrivial_combination" in residual_types:
        combo_f1 = float(metrics["per_class"]["nontrivial_combination"]["f1"])
    return float(
        0.60 * metrics.get("macro_f1", 0.0)
        + 0.30 * metrics.get("balanced_accuracy", 0.0)
        + 0.10 * combo_f1
    )


def residual_routing_constraints_ok(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    residual_types: list[str],
    *,
    minor_recall_min: float,
    combination_recall_min: float,
    max_pred_gold_ratio: float,
    max_combination_pred_gold_ratio: float,
) -> bool:
    pred_counts = residual_type_pred_counts(y_pred, residual_types)
    true_counts = residual_type_pred_counts(y_true, residual_types)
    if "minor_variant" in residual_types:
        minor = metrics["per_class"]["minor_variant"]
        if true_counts["minor_variant"] > 0 and float(minor["recall"]) < minor_recall_min:
            return False
    if "nontrivial_combination" in residual_types:
        combo = metrics["per_class"]["nontrivial_combination"]
        if true_counts["nontrivial_combination"] > 0 and float(combo["recall"]) < combination_recall_min:
            return False
        gold_combo = true_counts["nontrivial_combination"]
        if gold_combo > 0 and pred_counts["nontrivial_combination"] > max_combination_pred_gold_ratio * gold_combo:
            return False
    for label in residual_types:
        gold = true_counts[label]
        pred = pred_counts[label]
        if gold > 0 and pred == 0:
            return False
        if gold > 0 and pred > max_pred_gold_ratio * gold:
            return False
        if gold == 0 and pred > 0:
            return False
    return True


def calibrate_residual_routing(
    y_true: np.ndarray,
    type_probabilities: np.ndarray,
    substantive_probability: np.ndarray,
    innovation_family_probabilities: np.ndarray,
    combination_probability: np.ndarray,
    residual_types: list[str],
) -> dict[str, Any]:
    substantive_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    combination_thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    best: dict[str, Any] | None = None
    best_unconstrained: dict[str, Any] | None = None
    for substantive_threshold in substantive_thresholds:
        for combination_threshold in combination_thresholds:
            pred, routed_combo, joint_score = routed_residual_predictions(
                type_probabilities,
                substantive_probability,
                innovation_family_probabilities,
                combination_probability,
                residual_types,
                substantive_threshold=substantive_threshold,
                combination_threshold=combination_threshold,
            )
            metrics = classification_metrics(y_true, pred, residual_types)
            score = residual_routing_objective(metrics, residual_types)
            constraints_ok = residual_routing_constraints_ok(
                y_true,
                pred,
                metrics,
                residual_types,
                minor_recall_min=0.25,
                combination_recall_min=0.25,
                max_pred_gold_ratio=3.0,
                max_combination_pred_gold_ratio=2.0,
            )
            candidate = {
                "substantive_threshold": float(substantive_threshold),
                "combination_threshold": float(combination_threshold),
                "score": round(float(score), 6),
                "constraints_ok": bool(constraints_ok),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "minor_recall": metrics["per_class"].get("minor_variant", {}).get("recall", 0.0),
                "combination_recall": metrics["per_class"].get("nontrivial_combination", {}).get("recall", 0.0),
                "combination_f1": metrics["per_class"].get("nontrivial_combination", {}).get("f1", 0.0),
                "pred_distribution": metrics.get("pred_distribution", {}),
                "combination_pred_count": int(routed_combo.sum()),
                "combination_joint_score_mean": round(float(joint_score.mean()), 6) if len(joint_score) else 0.0,
            }
            if best_unconstrained is None or score > float(best_unconstrained["score"]):
                best_unconstrained = candidate
            if constraints_ok and (best is None or score > float(best["score"])):
                best = candidate
    out = best if best is not None else best_unconstrained
    if out is None:
        return {
            "substantive_threshold": 0.50,
            "combination_threshold": 0.08,
            "score": 0.0,
            "constraints_ok": False,
        }
    return {**out, "used_fallback_unconstrained": best is None}


def calibrate_novelty_score_cuts(
    y_true: np.ndarray,
    predicted_score: np.ndarray,
    predicted_evidence: np.ndarray,
) -> dict[str, Any]:
    if predicted_score.size < 8:
        return {"cuts": list(NOVELTY_FIXED_CUTS), "accuracy": 0.0, "fitted": False}

    candidates = np.unique(np.quantile(predicted_score, np.linspace(0.01, 0.99, 99)))

    def search(evidence: np.ndarray | None, gate: float | None) -> tuple[list[float], float]:
        cuts = [float(c) for c in np.quantile(predicted_score, [0.25, 0.50, 0.75])]
        best = float(
            (novelty_labels_from_score(predicted_score, evidence, cuts, evidence_gate=gate) == y_true).mean()
        )
        for _ in range(12):
            improved = False
            for position in range(3):
                for candidate in candidates:
                    trial = list(cuts)
                    trial[position] = float(candidate)
                    if not (trial[0] <= trial[1] <= trial[2]):
                        continue
                    accuracy = float(
                        (
                            novelty_labels_from_score(predicted_score, evidence, trial, evidence_gate=gate)
                            == y_true
                        ).mean()
                    )
                    if accuracy > best + 1e-12:
                        best, cuts, improved = accuracy, trial, True
            if not improved:
                break
        return cuts, best

    gate_candidates = [float(NOVELTY_EVIDENCE_GATE)]
    if predicted_evidence.size:
        gate_candidates += [
            float(g) for g in np.unique(np.quantile(predicted_evidence, np.linspace(0.02, 0.98, 49)))
        ]
    gated_cuts, gated_accuracy, gated_threshold = None, -1.0, float(NOVELTY_EVIDENCE_GATE)
    for gate in gate_candidates:
        cuts, accuracy = search(predicted_evidence, gate)
        if accuracy > gated_accuracy:
            gated_cuts, gated_accuracy, gated_threshold = cuts, accuracy, gate

    plain_cuts, plain_accuracy = search(None, None)
    use_gate = gated_cuts is not None and gated_accuracy >= plain_accuracy
    return {
        "cuts": gated_cuts if use_gate else plain_cuts,
        "accuracy": round(float(max(gated_accuracy, plain_accuracy)), 6),
        "use_evidence_gate": bool(use_gate),
        "evidence_gate": round(float(gated_threshold), 6) if use_gate else None,
        "gated": {
            "cuts": [round(c, 6) for c in (gated_cuts or [])],
            "accuracy": round(float(gated_accuracy), 6),
            "evidence_gate": round(float(gated_threshold), 6),
        },
        "ungated": {"cuts": [round(c, 6) for c in plain_cuts], "accuracy": round(float(plain_accuracy), 6)},
        "dataset_cuts_accuracy": round(
            float(
                (
                    novelty_labels_from_score(predicted_score, predicted_evidence, NOVELTY_FIXED_CUTS)
                    == y_true
                ).mean()
            ),
            6,
        ),
        "fitted": True,
    }


def novelty_fusion_candidates() -> list[tuple[float, float, float]]:
    candidates: set[tuple[float, float, float]] = {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.50, 0.35, 0.15),
    }
    for direct_weight in [0.40, 0.50, 0.60, 0.70]:
        for ordinal_weight in [0.20, 0.30, 0.40, 0.50]:
            for residual_weight in [0.00, 0.10, 0.20, 0.30]:
                total = direct_weight + ordinal_weight + residual_weight
                if abs(total - 1.0) <= 1e-6:
                    candidates.add((direct_weight, ordinal_weight, residual_weight))
    return sorted(candidates)


def fused_novelty_decision(
    direct_probabilities: np.ndarray,
    ordinal_probabilities: np.ndarray,
    residual_probabilities: np.ndarray,
    moderate_high_probability: np.ndarray,
    *,
    fusion_weights: tuple[float, float, float],
    moderate_high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(fusion_weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    weights = weights / max(1e-12, float(weights.sum()))
    fused = (
        weights[0] * direct_probabilities
        + weights[1] * ordinal_probabilities
        + weights[2] * residual_probabilities
    )
    fused = fused / np.clip(fused.sum(axis=1, keepdims=True), 1e-12, None)
    pred = fused.argmax(axis=1).astype(np.int64)
    high_joint_score = (
        0.40 * direct_probabilities[:, NOVELTY_TO_ID["high_novelty"]]
        + 0.30 * ordinal_probabilities[:, NOVELTY_TO_ID["high_novelty"]]
        + 0.30 * moderate_high_probability
    )
    upgrade = (pred == NOVELTY_TO_ID["moderate_novelty"]) & (high_joint_score >= moderate_high_threshold)
    pred = pred.copy()
    pred[upgrade] = NOVELTY_TO_ID["high_novelty"]
    return pred, fused.astype(np.float32), high_joint_score.astype(np.float32)


def novelty_objective(metrics: dict[str, Any]) -> float:
    return float(
        0.45 * metrics.get("qwk", 0.0)
        + 0.35 * metrics.get("macro_f1", 0.0)
        + 0.20 * metrics.get("balanced_accuracy", 0.0)
    )


def calibrate_novelty_fusion(
    y_true: np.ndarray,
    direct_probabilities: np.ndarray,
    ordinal_probabilities: np.ndarray,
    residual_probabilities: np.ndarray,
    moderate_high_probability: np.ndarray,
) -> dict[str, Any]:
    moderate_high_thresholds = [0.20, 0.22, 0.24, 0.26, 0.2688, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]
    best: dict[str, Any] | None = None
    for weights in novelty_fusion_candidates():
        for threshold in moderate_high_thresholds:
            pred, _, high_joint_score = fused_novelty_decision(
                direct_probabilities,
                ordinal_probabilities,
                residual_probabilities,
                moderate_high_probability,
                fusion_weights=weights,
                moderate_high_threshold=threshold,
            )
            metrics = classification_metrics(y_true, pred, NOVELTY_LABELS, ordinal=True)
            score = novelty_objective(metrics)
            candidate = {
                "direct_weight": float(weights[0]),
                "ordinal_weight": float(weights[1]),
                "residual_weight": float(weights[2]),
                "moderate_high_threshold": float(threshold),
                "score": round(float(score), 6),
                "qwk": metrics.get("qwk", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "high_recall": metrics["per_class"]["high_novelty"]["recall"],
                "high_f1": metrics["per_class"]["high_novelty"]["f1"],
                "weak_recall": metrics["per_class"]["weak_novelty"]["recall"],
                "moderate_recall": metrics["per_class"]["moderate_novelty"]["recall"],
                "pred_distribution": metrics.get("pred_distribution", {}),
                "high_joint_score_mean": round(float(high_joint_score.mean()), 6) if len(high_joint_score) else 0.0,
            }
            if best is None or score > float(best["score"]):
                best = candidate
    if best is None:
        return {
            "direct_weight": 0.50,
            "ordinal_weight": 0.35,
            "residual_weight": 0.15,
            "moderate_high_threshold": 0.2688,
            "score": 0.0,
        }
    return best


def semantic_metrics(
    semantic_attention_batches: list[np.ndarray],
    coverage_batches: list[np.ndarray],
    query_tensor: torch.Tensor,
    active_batches: list[np.ndarray] | None = None,
    semantic_unit_batches: list[np.ndarray] | None = None,
    slot_assignment_batches: list[np.ndarray] | None = None,
    target_mask_batches: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        normalized = F.normalize(query_tensor.detach().cpu().float(), dim=-1)
        cosine = (normalized @ normalized.T).numpy()
    off_diag = cosine[~np.eye(cosine.shape[0], dtype=bool)] if cosine.size else np.asarray([])
    attentions = np.concatenate(semantic_attention_batches, axis=0) if semantic_attention_batches else np.zeros((0, 0, 0))
    coverages = np.concatenate(coverage_batches, axis=0) if coverage_batches else np.zeros((0, 0))
    active_probs = np.concatenate(active_batches, axis=0) if active_batches else np.zeros((0, 0))
    semantic_units = (
        np.concatenate(semantic_unit_batches, axis=0)
        if semantic_unit_batches
        else np.zeros((0, 0, 0))
    )
    slot_assignments = (
        np.concatenate(slot_assignment_batches, axis=0)
        if slot_assignment_batches
        else np.zeros((0, 0, 0))
    )
    target_masks = (
        np.concatenate(target_mask_batches, axis=0).astype(bool)
        if target_mask_batches
        else np.zeros((0, 0), dtype=bool)
    )
    if attentions.size:
        normalized_entropies = []
        overlaps = []
        top_token_unique_ratios = []
        for row_idx, row in enumerate(attentions):
            valid_mask = (
                target_masks[row_idx]
                if row_idx < len(target_masks)
                else np.ones(row.shape[-1], dtype=bool)
            )
            valid_count = max(2, int(valid_mask.sum()))
            entropy = -(row * np.log(np.clip(row, 1e-8, 1.0))).sum(axis=-1)
            normalized_entropies.extend((entropy / math.log(valid_count)).tolist())
            for i in range(row.shape[0]):
                for j in range(i + 1, row.shape[0]):
                    overlaps.append(float(np.minimum(row[i], row[j]).sum()))
            top_k = min(3, int(valid_mask.sum()))
            if top_k > 0:
                selected: list[int] = []
                for unit_attention in row:
                    masked_attention = np.where(valid_mask, unit_attention, -1.0)
                    ranked_positions = np.argsort(-masked_attention)
                    positive_positions = [
                        int(position)
                        for position in ranked_positions
                        if masked_attention[position] > 1e-8
                    ][:top_k]
                    selected.extend(positive_positions)
                if selected:
                    top_token_unique_ratios.append(
                        len(set(selected)) / len(selected)
                    )
        attention_entropy = float(np.mean(normalized_entropies)) if normalized_entropies else 0.0
        token_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        top_token_unique_ratio = (
            float(np.mean(top_token_unique_ratios))
            if top_token_unique_ratios
            else 0.0
        )
    else:
        attention_entropy = 0.0
        token_overlap = 0.0
        top_token_unique_ratio = 0.0
    if slot_assignments.size:
        slot_entropy_values = []
        slot_confidence_values = []
        slot_mass_ratios = []
        empty_slot_values = []
        slot_mass_cv_values = []
        effective_slot_count_values = []
        for row_idx, assignment in enumerate(slot_assignments):
            valid_mask = (
                target_masks[row_idx]
                if row_idx < len(target_masks)
                else np.ones(assignment.shape[-1], dtype=bool)
            )
            if not valid_mask.any():
                continue
            probabilities = assignment[:, valid_mask]
            probabilities = probabilities / np.clip(
                probabilities.sum(axis=0, keepdims=True),
                1e-8,
                None,
            )
            entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=0)
            entropy = entropy / math.log(max(2, probabilities.shape[0]))
            slot_entropy_values.extend(entropy.tolist())
            slot_confidence_values.extend(probabilities.max(axis=0).tolist())
            slot_mass = probabilities.sum(axis=1)
            expected_slot_mass = max(
                1e-8,
                probabilities.shape[1] / max(1, probabilities.shape[0]),
            )
            mass_ratio = slot_mass / expected_slot_mass
            slot_mass_ratios.append(mass_ratio)
            empty_slot_values.extend(
                (mass_ratio < 0.10).astype(np.float32).tolist()
            )
            slot_mass_cv_values.append(
                float(slot_mass.std() / max(1e-8, slot_mass.mean()))
            )
            slot_usage = slot_mass / max(1e-8, float(slot_mass.sum()))
            effective_slot_count_values.append(
                float(
                    np.exp(
                        -(
                            slot_usage
                            * np.log(np.clip(slot_usage, 1e-8, 1.0))
                        ).sum()
                    )
                )
            )
        slot_assignment_entropy = (
            float(np.mean(slot_entropy_values))
            if slot_entropy_values
            else 0.0
        )
        slot_assignment_confidence = (
            float(np.mean(slot_confidence_values))
            if slot_confidence_values
            else 0.0
        )
        slot_mass_ratio_array = (
            np.stack(slot_mass_ratios)
            if slot_mass_ratios
            else np.zeros((0, slot_assignments.shape[1]))
        )
        empty_slot_ratio = (
            float(np.mean(empty_slot_values))
            if empty_slot_values
            else 0.0
        )
        minimum_slot_mass_ratio = (
            float(slot_mass_ratio_array.min(axis=1).mean())
            if len(slot_mass_ratio_array)
            else 0.0
        )
        slot_mass_coefficient_of_variation = (
            float(np.mean(slot_mass_cv_values))
            if slot_mass_cv_values
            else 0.0
        )
        effective_slot_count = (
            float(np.mean(effective_slot_count_values))
            if effective_slot_count_values
            else 0.0
        )
        effective_slot_ratio = effective_slot_count / max(
            1,
            slot_assignments.shape[1],
        )
    else:
        slot_assignment_entropy = 0.0
        slot_assignment_confidence = 0.0
        slot_mass_ratio_array = np.zeros((0, 0))
        empty_slot_ratio = 0.0
        minimum_slot_mass_ratio = 0.0
        slot_mass_coefficient_of_variation = 0.0
        effective_slot_count = 0.0
        effective_slot_ratio = 0.0
    if semantic_units.size and semantic_units.shape[1] > 1:
        norms = np.linalg.norm(semantic_units, axis=-1, keepdims=True)
        normalized_units = semantic_units / np.clip(norms, 1e-8, None)
        unit_cosine = np.matmul(normalized_units, np.swapaxes(normalized_units, 1, 2))
        pair_mask = np.broadcast_to(
            ~np.eye(unit_cosine.shape[-1], dtype=bool),
            unit_cosine.shape,
        ).copy()
        if slot_mass_ratio_array.shape[:2] == semantic_units.shape[:2]:
            active_slots = slot_mass_ratio_array >= 0.10
            pair_mask &= active_slots[:, :, None] & active_slots[:, None, :]
        unit_off_diagonal = unit_cosine[pair_mask]
        unit_pairwise_cosine = (
            float(unit_off_diagonal.mean())
            if unit_off_diagonal.size
            else 0.0
        )
    else:
        unit_pairwise_cosine = 0.0
    if len(active_probs):
        usage = active_probs.mean(axis=0)
        usage = usage / max(1e-8, float(usage.sum()))
        usage_entropy = float(-(usage * np.log(np.clip(usage, 1e-8, 1.0))).sum())
        effective_active_query_count = float(math.exp(usage_entropy))
        effective_active_query_ratio = effective_active_query_count / max(1, active_probs.shape[1])
    else:
        usage_entropy = 0.0
        effective_active_query_count = 0.0
        effective_active_query_ratio = 0.0
    coverage_variance = float(coverages.var(axis=1).mean()) if len(coverages) else 0.0
    semantic_health_score = float(
        0.20 * (1.0 - np.clip(token_overlap, 0.0, 1.0))
        + 0.15 * np.clip(top_token_unique_ratio, 0.0, 1.0)
        + 0.15 * np.clip(slot_assignment_confidence, 0.0, 1.0)
        + 0.10 * (1.0 - np.clip(unit_pairwise_cosine, 0.0, 1.0))
        + 0.15 * np.clip(coverage_variance / 0.005, 0.0, 1.0)
        + 0.15 * (1.0 - np.clip(empty_slot_ratio, 0.0, 1.0))
        + 0.10 * np.clip(minimum_slot_mass_ratio, 0.0, 1.0)
    )
    semantic_constraints_ok = bool(
        token_overlap <= 0.80
        and top_token_unique_ratio >= 0.50
        and slot_assignment_confidence >= 0.45
        and unit_pairwise_cosine <= 0.90
        and coverage_variance >= 0.0005
        and empty_slot_ratio <= 0.05
        and minimum_slot_mass_ratio >= 0.25
        and effective_slot_ratio >= 0.75
    )
    return {
        "query_pairwise_cosine": round(float(off_diag.mean()), 4) if len(off_diag) else 0.0,
        "query_pairwise_abs_cosine": round(float(np.abs(off_diag).mean()), 4) if len(off_diag) else 0.0,
        "query_attention_entropy": round(attention_entropy, 4),
        "unit_coverage_variance": round(coverage_variance, 6),
        "active_query_count": round(float((coverages > 0.10).sum(axis=1).mean()), 4) if len(coverages) else 0.0,
        "effective_active_query_count": round(effective_active_query_count, 4),
        "effective_active_query_ratio": round(effective_active_query_ratio, 4),
        "active_unit_ratio": round(float((active_probs >= 0.5).mean()), 4) if len(active_probs) else 0.0,
        "mean_active_probability": round(float(active_probs.mean()), 4) if len(active_probs) else 0.0,
        "unit_usage_entropy": round(usage_entropy, 4),
        "query_token_overlap": round(token_overlap, 4),
        "top_token_unique_ratio": round(top_token_unique_ratio, 4),
        "slot_assignment_entropy": round(slot_assignment_entropy, 4),
        "slot_assignment_confidence": round(slot_assignment_confidence, 4),
        "empty_slot_ratio": round(empty_slot_ratio, 4),
        "minimum_slot_mass_ratio": round(minimum_slot_mass_ratio, 4),
        "slot_mass_coefficient_of_variation": round(
            slot_mass_coefficient_of_variation,
            4,
        ),
        "effective_slot_count": round(effective_slot_count, 4),
        "effective_slot_ratio": round(effective_slot_ratio, 4),
        "unit_representation_pairwise_cosine": round(unit_pairwise_cosine, 4),
        "semantic_health_score": round(semantic_health_score, 4),
        "semantic_constraints_ok": semantic_constraints_ok,
    }


def residual_selection_score(metrics: dict[str, Any]) -> float:
    return round(float(
        0.35 * metrics["residual_type"]["macro_f1"]
        + 0.15 * metrics["residual_type"]["balanced_accuracy"]
        + 0.20 * metrics["substantive_delta_score"]["spearman"]
        + 0.15 * metrics["evidence_sufficiency"]["spearman"]
        + 0.15 * metrics["estimated_unit_coverage_vs_predicted_joint"]["spearman"]
    ), 6)


def residual_selection_constraints_ok(metrics: dict[str, Any]) -> bool:
    residual = metrics["residual_type"]
    per_class = residual["per_class"]
    pred_distribution = residual.get("pred_distribution", {})
    true_distribution = residual.get("true_distribution", {})
    if "minor_variant" in per_class and true_distribution.get("minor_variant", 0) > 0:
        if float(per_class["minor_variant"]["recall"]) < 0.20:
            return False
    if "nontrivial_combination" in per_class and true_distribution.get("nontrivial_combination", 0) > 0:
        if float(per_class["nontrivial_combination"]["recall"]) < 0.20:
            return False
    for label, gold_count in true_distribution.items():
        pred_count = int(pred_distribution.get(label, 0))
        if int(gold_count) > 0 and pred_count == 0:
            return False
        if int(gold_count) > 0 and pred_count > 3.0 * int(gold_count):
            return False
    for label, pred_count in pred_distribution.items():
        if int(true_distribution.get(label, 0)) == 0 and int(pred_count) > 0:
            return False
    return True


def novelty_selection_score(metrics: dict[str, Any]) -> tuple[float, bool]:
    novelty = metrics.get("novelty_final", metrics["novelty_ordinal"])
    score = float(
        0.45 * novelty.get("qwk", 0.0)
        + 0.35 * novelty.get("macro_f1", 0.0)
        + 0.20 * novelty.get("balanced_accuracy", 0.0)
    )
    per_class = novelty["per_class"]
    pred_distribution = novelty.get("pred_distribution", {})
    constraints_ok = (
        float(per_class["high_novelty"]["recall"]) >= 0.10
        and float(per_class["high_novelty"]["f1"]) >= 0.10
        and float(per_class["weak_novelty"]["recall"]) >= 0.20
        and float(per_class["moderate_novelty"]["recall"]) >= 0.40
        and all(int(pred_distribution.get(label, 0)) > 0 for label in NOVELTY_LABELS)
    )
    return round(score, 6), bool(constraints_ok)


def total_selection_score(metrics: dict[str, Any]) -> float:
    residual_score = metrics.get("residual_selection_score")
    if residual_score is None:
        residual_score = residual_selection_score(metrics)
    novelty_score_value, _ = novelty_selection_score(metrics)
    return round(float(0.45 * residual_score + 0.55 * novelty_score_value), 6)


def novelty_label_from_score(
    score: float,
    evidence: float,
    uncertainty: float,
    mostly_covered_probability: float = 0.0,
) -> int:
    if mostly_covered_probability >= 0.70 and evidence >= 0.55 and uncertainty <= 0.55:
        if score >= 0.48:
            return NOVELTY_TO_ID["high_novelty"]
        if score >= 0.28:
            return NOVELTY_TO_ID["moderate_novelty"]
        if score >= 0.12:
            return NOVELTY_TO_ID["weak_novelty"]
        return NOVELTY_TO_ID["low_novelty"]
    if evidence < 0.55 or uncertainty > 0.70:
        if score >= 0.30:
            return NOVELTY_TO_ID["moderate_novelty"]
        if score >= 0.12:
            return NOVELTY_TO_ID["weak_novelty"]
        return NOVELTY_TO_ID["low_novelty"]
    if score >= 0.42:
        return NOVELTY_TO_ID["high_novelty"]
    if score >= 0.24:
        return NOVELTY_TO_ID["moderate_novelty"]
    if score >= 0.10:
        return NOVELTY_TO_ID["weak_novelty"]
    return NOVELTY_TO_ID["low_novelty"]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def score_to_joint_probability_tensor(score: torch.Tensor) -> torch.Tensor:
    anchors = torch.linspace(0.0, 1.0, len(JOINT_LABELS), device=score.device, dtype=score.dtype)
    logits = -((score.unsqueeze(-1).clamp(0.0, 1.0) - anchors) ** 2) / (2.0 * 0.18 * 0.18)
    return torch.softmax(logits, dim=-1)


def joint_entropy_from_probabilities(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    normalized = entropy / math.log(len(JOINT_LABELS))
    return entropy, normalized.clamp(0.0, 1.0)


def gold_coverage_input_ratio(args: argparse.Namespace, epoch: int) -> float:
    stage_epochs = max(1, int(args.gold_coverage_mix_stage_epochs))
    if epoch <= stage_epochs:
        return clamp01(args.gold_coverage_input_ratio_start)
    if epoch <= stage_epochs * 2:
        return clamp01(args.gold_coverage_input_ratio_mid)
    return clamp01(args.gold_coverage_input_ratio_end)


def maybe_mix_coverage_inputs(batch: dict[str, Any], args: argparse.Namespace, epoch: int) -> dict[str, Any]:
    ratio = gold_coverage_input_ratio(args, epoch)
    if ratio <= 0.0:
        return batch
    gold_score = batch["gold_joint_coverage_score"].clamp(0.0, 1.0)
    mix_mask = torch.rand(gold_score.shape, device=gold_score.device) < ratio
    if not bool(mix_mask.any()):
        return batch

    mixed = dict(batch)
    global_features = batch["global_features"].clone()
    gold_probs = score_to_joint_probability_tensor(gold_score)
    gold_entropy, gold_norm_entropy = joint_entropy_from_probabilities(gold_probs)
    pair_union = batch["pair_union_score"].to(gold_score.device)

    feature_values = {
        "joint_evidence_score": gold_score,
        "joint_expected_score": expected_score_from_tensor(gold_probs),
        "joint_entropy": gold_entropy,
        "joint_normalized_entropy": gold_norm_entropy,
        "has_joint_prediction": torch.ones_like(gold_score),
        "joint_minus_pair_union": gold_score - pair_union,
        "p_not_covered": gold_probs[:, 0],
        "p_weakly_covered": gold_probs[:, 1],
        "p_partially_covered": gold_probs[:, 2],
        "p_mostly_covered": gold_probs[:, 3],
    }
    for name, values in feature_values.items():
        col = GLOBAL_FEATURE_NAMES.index(name)
        global_features[:, col] = torch.where(mix_mask, values.to(global_features.dtype), global_features[:, col])

    mixed["global_features"] = global_features
    mixed["joint_evidence_score"] = torch.where(mix_mask, gold_score, batch["joint_evidence_score"])
    mixed["joint_normalized_entropy"] = torch.where(mix_mask, gold_norm_entropy, batch["joint_normalized_entropy"])
    return mixed


def expected_score_from_tensor(probs: torch.Tensor) -> torch.Tensor:
    scale = torch.linspace(0.0, 1.0, len(JOINT_LABELS), device=probs.device, dtype=probs.dtype)
    return (probs * scale).sum(dim=-1)


def amp_dtype_from_arg(device: torch.device, amp_dtype: str) -> torch.dtype:
    if amp_dtype in {"bf16", "bfloat16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cpu":
            return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype) -> contextlib.AbstractContextManager:
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(
    model: EGRDResidualPredictor,
    loader: DataLoader,
    device: torch.device,
    residual_types: list[str],
    tokenizer: Any,
    *,
    surface_penalty_lambda: float,
    substantive_threshold: float = 0.5,
    combination_threshold: float = 0.08,
    novelty_fusion_weights: tuple[float, float, float] = (0.50, 0.35, 0.15),
    moderate_high_threshold: float = 0.2688,
    novelty_score_cuts: Sequence[float] = NOVELTY_FIXED_CUTS,
    novelty_score_evidence_gate: float | None = NOVELTY_EVIDENCE_GATE,
    semantic_top_token_count: int = 6,
    embedding_sink: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    y_sub: list[float] = []
    p_sub: list[float] = []
    y_surface: list[float] = []
    p_surface: list[float] = []
    y_evidence: list[float] = []
    p_evidence: list[float] = []
    y_type: list[int] = []
    p_type: list[int] = []
    y_coarse: list[int] = []
    p_coarse: list[int] = []
    y_substantive_type: list[int] = []
    p_substantive_type: list[int] = []
    s_substantive_type: list[float] = []
    y_innovation_family: list[int] = []
    p_innovation_family: list[int] = []
    y_combination: list[float] = []
    p_combination: list[int] = []
    p_combination_raw: list[int] = []
    s_combination: list[float] = []
    s_combination_raw: list[float] = []
    y_novelty: list[int] = []
    p_novelty_direct_aux: list[int] = []
    p_novelty_ordinal: list[int] = []
    p_novelty_residual_branch: list[int] = []
    p_novelty_final: list[int] = []
    p_novelty_rule: list[int] = []
    y_moderate_high: list[int] = []
    p_moderate_high: list[int] = []
    s_moderate_high: list[float] = []
    raw_novelty_scores: list[float] = []
    composed_novelty_scores: list[float] = []
    y_novelty_score: list[float] = []
    p_novelty_score: list[float] = []
    y_joint_input: list[float] = []
    y_joint_gold: list[float] = []
    p_joint_est: list[float] = []
    semantic_attention_batches: list[np.ndarray] = []
    coverage_batches: list[np.ndarray] = []
    active_batches: list[np.ndarray] = []
    semantic_unit_batches: list[np.ndarray] = []
    slot_assignment_batches: list[np.ndarray] = []
    semantic_target_mask_batches: list[np.ndarray] = []
    type_probability_batches: list[np.ndarray] = []
    substantive_probability_batches: list[np.ndarray] = []
    family_probability_batches: list[np.ndarray] = []
    combination_probability_batches: list[np.ndarray] = []
    direct_probability_batches: list[np.ndarray] = []
    ordinal_probability_batches: list[np.ndarray] = []
    residual_novelty_probability_batches: list[np.ndarray] = []
    moderate_high_probability_batches: list[np.ndarray] = []
    fusion_gate_batches: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        device_batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
            token_type_ids=device_batch.get("token_type_ids"),
            target_token_mask=device_batch["target_token_mask"],
            pair_mask=device_batch["pair_mask"],
            pair_features=device_batch["pair_features"],
            global_features=device_batch["global_features"],
            aspect_id=device_batch["aspect_id"],
            contribution_id=device_batch["contribution_id"],
            task_id=device_batch["task_id"],
            joint_normalized_entropy=device_batch["joint_normalized_entropy"],
        )
        if embedding_sink is not None:
            for name in ("repr_unit_residual", "repr_interaction_residual", "repr_covered",
                         "repr_target", "repr_global_context", "repr_fused"):
                embedding_sink.setdefault(name, []).append(
                    outputs[name].detach().float().cpu().numpy()
                )
            embedding_sink.setdefault("target_idea_id", []).extend(batch["target_idea_id"])
        sub = outputs["substantive_score"].detach().cpu().numpy()
        surface = outputs["surface_score"].detach().cpu().numpy()
        evidence = outputs["evidence_sufficiency"].detach().cpu().numpy()
        uncertainty = outputs["prediction_uncertainty"].detach().cpu().numpy()
        type_probs = outputs["type_probabilities"].detach().cpu().numpy()
        coarse_probs = torch.softmax(outputs["coarse_type_logits"].detach().cpu().float(), dim=-1).numpy()
        substantive_prob = torch.sigmoid(outputs["minor_substantive_logits"].detach().cpu().float().squeeze(-1)).numpy()
        family_probs = torch.softmax(outputs["innovation_family_logits"].detach().cpu().float(), dim=-1).numpy()
        combo_prob = outputs["combination_probability"].detach().cpu().numpy()
        direct_probs = outputs["novelty_direct_probabilities"].detach().cpu().numpy()
        ordinal_probs = outputs["novelty_ordinal_probabilities"].detach().cpu().numpy()
        residual_novelty_probs = outputs["residual_novelty_probabilities"].detach().cpu().numpy()
        fusion_gates = outputs["novelty_fusion_gates"].detach().cpu().numpy()
        mh_prob = outputs["moderate_high_probability"].detach().cpu().numpy()
        novelty_probs = outputs["novelty_probabilities"].detach().cpu().numpy()
        estimated_joint = outputs["estimated_joint_coverage"].detach().cpu().numpy()
        coverage_scores = outputs["semantic_unit_coverage_scores"].detach().cpu().numpy()
        residual_scores = outputs["semantic_unit_residual_scores"].detach().cpu().numpy()
        unit_active = outputs["unit_active_probability"].detach().cpu().numpy()
        semantic_slot_mass = outputs["semantic_slot_mass"].detach().cpu().numpy()
        coverage_gate = outputs["coverage_gate"].detach().cpu().numpy()
        weighted_support = outputs["weighted_support"].detach().cpu().numpy()
        pair_weights = outputs["pair_weights"].detach().cpu().numpy()
        interaction_scores = outputs["interaction_residual_scores"].detach().cpu().numpy()
        joint_input = batch["joint_evidence_score"].detach().cpu().numpy()
        mostly_covered = batch["global_features"][:, GLOBAL_FEATURE_NAMES.index("p_mostly_covered")].detach().cpu().numpy()
        gold_joint = batch["gold_joint_coverage_score"].detach().cpu().numpy()
        novelty_score = (1.0 - joint_input) * sub * (1.0 - surface_penalty_lambda * surface)
        novelty_score = np.clip(novelty_score, 0.0, 1.0)
        composed_score = np.clip(
            (1.0 - joint_input) * sub * evidence * (1.0 - surface_penalty_lambda * surface), 0.0, 1.0
        )
        head_score = outputs["predicted_novelty_score"].detach().cpu().numpy()
        gold_score = batch["gold_novelty_score"].detach().cpu().numpy()
        pred_rule = np.asarray(
            [
                novelty_label_from_score(
                    float(novelty_score[idx]),
                    float(evidence[idx]),
                    float(uncertainty[idx]),
                    float(mostly_covered[idx]),
                )
                for idx in range(len(sub))
            ],
            dtype=np.int64,
        )
        pred_direct_aux = direct_probs.argmax(axis=1).astype(np.int64)
        pred_ordinal = ordinal_probs.argmax(axis=1).astype(np.int64)
        pred_residual_branch = residual_novelty_probs.argmax(axis=1).astype(np.int64)
        pred_final, novelty_probs, high_joint_score = fused_novelty_decision(
            direct_probs,
            ordinal_probs,
            residual_novelty_probs,
            mh_prob,
            fusion_weights=novelty_fusion_weights,
            moderate_high_threshold=moderate_high_threshold,
        )
        pred_type, pred_combo, combination_joint_score = routed_residual_predictions(
            type_probs,
            substantive_prob,
            family_probs,
            combo_prob,
            residual_types,
            substantive_threshold=substantive_threshold,
            combination_threshold=combination_threshold,
        )
        pred_coarse = np.asarray(
            [COARSE_TO_ID[coarse_type_for(residual_types[int(type_id)])] for type_id in pred_type],
            dtype=np.int64,
        )
        pred_combo_raw = (combo_prob >= 0.5).astype(np.int64)
        gold_type = batch["residual_type_id"].detach().cpu().numpy().astype(np.int64)
        gold_coarse = batch["coarse_type_id"].detach().cpu().numpy().astype(np.int64)
        gold_substantive = batch["is_substantive_type"].detach().cpu().numpy().astype(np.int64)
        gold_family = batch["innovation_family_id"].detach().cpu().numpy().astype(np.int64)
        pred_family = family_probs.argmax(axis=1).astype(np.int64)
        gold_combo = batch["is_nontrivial_combination"].detach().cpu().numpy()
        mh_mask = batch["moderate_high_mask"].detach().cpu().numpy() > 0.5
        gold_mh = batch["moderate_high_target"].detach().cpu().numpy()
        semantic_attention_np = outputs["semantic_attention"].detach().cpu().numpy()
        semantic_units_np = outputs["semantic_units"].detach().cpu().numpy()
        slot_assignment_np = outputs["semantic_token_assignment"].detach().cpu().numpy()
        semantic_target_mask_np = outputs["semantic_target_mask"].detach().cpu().numpy().astype(bool)
        input_ids_np = batch["input_ids"].detach().cpu().numpy()
        pair_mask_np = batch["pair_mask"].detach().cpu().numpy().astype(bool)
        special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        semantic_attention_batches.append(semantic_attention_np)
        coverage_batches.append(coverage_scores)
        active_batches.append(unit_active)
        semantic_unit_batches.append(semantic_units_np)
        slot_assignment_batches.append(slot_assignment_np)
        semantic_target_mask_batches.append(semantic_target_mask_np)
        type_probability_batches.append(type_probs)
        substantive_probability_batches.append(substantive_prob)
        family_probability_batches.append(family_probs)
        combination_probability_batches.append(combo_prob)
        direct_probability_batches.append(direct_probs)
        ordinal_probability_batches.append(ordinal_probs)
        residual_novelty_probability_batches.append(residual_novelty_probs)
        moderate_high_probability_batches.append(mh_prob)
        fusion_gate_batches.append(fusion_gates)

        y_sub.extend(batch["substantive_delta_score"].detach().cpu().numpy().tolist())
        p_sub.extend(sub.tolist())
        y_surface.extend(batch["surface_change_score"].detach().cpu().numpy().tolist())
        p_surface.extend(surface.tolist())
        y_evidence.extend(batch["evidence_sufficiency"].detach().cpu().numpy().tolist())
        p_evidence.extend(evidence.tolist())
        y_type.extend(gold_type.tolist())
        p_type.extend(pred_type.tolist())
        y_coarse.extend(gold_coarse.tolist())
        p_coarse.extend(pred_coarse.tolist())
        y_substantive_type.extend(gold_substantive.tolist())
        p_substantive_type.extend((substantive_prob >= substantive_threshold).astype(np.int64).tolist())
        s_substantive_type.extend(substantive_prob.tolist())
        substantive_mask_np = gold_substantive.astype(bool)
        y_innovation_family.extend(gold_family[substantive_mask_np].tolist())
        p_innovation_family.extend(pred_family[substantive_mask_np].tolist())
        y_combination.extend(gold_combo.tolist())
        p_combination.extend(pred_combo.tolist())
        p_combination_raw.extend(pred_combo_raw.tolist())
        s_combination.extend(combination_joint_score.tolist())
        s_combination_raw.extend(combo_prob.tolist())
        y_novelty.extend(batch["novelty_label_id"].detach().cpu().numpy().tolist())
        p_novelty_direct_aux.extend(pred_direct_aux.tolist())
        p_novelty_ordinal.extend(pred_ordinal.tolist())
        p_novelty_residual_branch.extend(pred_residual_branch.tolist())
        p_novelty_final.extend(pred_final.tolist())
        p_novelty_rule.extend(pred_rule.tolist())
        y_moderate_high.extend(gold_mh[mh_mask].astype(np.int64).tolist())
        p_moderate_high.extend((mh_prob[mh_mask] >= 0.5).astype(np.int64).tolist())
        s_moderate_high.extend(mh_prob[mh_mask].tolist())
        raw_novelty_scores.extend(novelty_score.tolist())
        composed_novelty_scores.extend(composed_score.tolist())
        p_novelty_score.extend(head_score.tolist())
        y_novelty_score.extend(gold_score.tolist())
        y_joint_input.extend(joint_input.tolist())
        y_joint_gold.extend(gold_joint.tolist())
        p_joint_est.extend(estimated_joint.tolist())

        for idx, target_id in enumerate(batch["target_idea_id"]):
            residual_rank = np.argsort(-residual_scores[idx])
            covered_rank = np.argsort(-coverage_scores[idx])
            unit_top_tokens = []
            if semantic_top_token_count > 0:
                valid_pair_indices = np.flatnonzero(pair_mask_np[idx])
                source_pair_idx = int(valid_pair_indices[0]) if len(valid_pair_indices) else 0
                source_ids = input_ids_np[idx, source_pair_idx]
                semantic_target_mask = semantic_target_mask_np[idx]
                for unit_id in range(coverage_scores.shape[1]):
                    unit_attention = semantic_attention_np[idx, unit_id].copy()
                    unit_attention[~semantic_target_mask] = -1.0
                    top_tokens = []
                    seen_token_positions: set[int] = set()
                    for token_position in np.argsort(-unit_attention):
                        if unit_attention[token_position] <= 0.0:
                            break
                        token_id = int(source_ids[token_position])
                        if token_id in special_token_ids or token_position in seen_token_positions:
                            continue
                        seen_token_positions.add(int(token_position))
                        top_tokens.append(
                            {
                                "token": tokenizer.convert_ids_to_tokens(token_id),
                                "token_position": int(token_position),
                                "attention": round(float(unit_attention[token_position]), 6),
                                "target_hidden_aggregation": "mean_across_valid_priors",
                            }
                        )
                        if len(top_tokens) >= semantic_top_token_count:
                            break
                    unit_top_tokens.append(
                        {
                            "semantic_unit_id": int(unit_id),
                            "active_probability": round(float(unit_active[idx, unit_id]), 6),
                            "coverage_score": round(float(coverage_scores[idx, unit_id]), 6),
                            "residual_score": round(float(residual_scores[idx, unit_id]), 6),
                            "top_target_tokens": top_tokens,
                        }
                    )
            residual_units = [
                {
                    "semantic_unit_id": int(unit_id),
                    "residual_score": round(float(residual_scores[idx, unit_id]), 6),
                    "coverage_score": round(float(coverage_scores[idx, unit_id]), 6),
                    "active_probability": round(float(unit_active[idx, unit_id]), 6),
                    "best_matching_prior_id": batch["prior_meta"][idx][int(weighted_support[idx, :, unit_id].argmax())]["prior_idea_id"],
                    "best_prior_support": round(float(weighted_support[idx, :, unit_id].max()), 6),
                }
                for unit_id in residual_rank[: min(3, len(residual_rank))]
            ]
            covered_units = [
                {
                    "semantic_unit_id": int(unit_id),
                    "coverage_score": round(float(coverage_scores[idx, unit_id]), 6),
                    "active_probability": round(float(unit_active[idx, unit_id]), 6),
                    "supporting_prior_ids": [
                        meta["prior_idea_id"]
                        for meta_idx, meta in enumerate(batch["prior_meta"][idx])
                        if meta["prior_idea_id"] and weighted_support[idx, meta_idx, unit_id] >= 0.25
                    ],
                }
                for unit_id in covered_rank[: min(3, len(covered_rank))]
            ]
            rows.append(
                {
                    "target_idea_id": target_id,
                    "gold_substantive_delta_score": round(float(batch["substantive_delta_score"][idx]), 6),
                    "pred_substantive_delta_score": round(float(sub[idx]), 6),
                    "gold_surface_change_score": round(float(batch["surface_change_score"][idx]), 6),
                    "pred_surface_change_score": round(float(surface[idx]), 6),
                    "gold_evidence_sufficiency": round(float(batch["evidence_sufficiency"][idx]), 6),
                    "pred_evidence_sufficiency": round(float(evidence[idx]), 6),
                    "pred_prediction_uncertainty": round(float(uncertainty[idx]), 6),
                    "joint_evidence_score": round(float(joint_input[idx]), 6),
                    "pair_union_score": round(float(batch["pair_union_score"][idx]), 6),
                    "estimated_joint_coverage_from_units": round(float(estimated_joint[idx]), 6),
                    "coverage_gate": round(float(coverage_gate[idx]), 6),
                    "pred_novelty_score": round(float(novelty_score[idx]), 6),
                    "gold_novelty_score_continuous": round(float(gold_score[idx]), 6),
                    "pred_novelty_score_head": round(float(head_score[idx]), 6),
                    "pred_novelty_score_composed": round(float(composed_score[idx]), 6),
                    "gold_residual_type": batch["gold_residual_type"][idx],
                    "pred_residual_type": residual_types[int(pred_type[idx])],
                    "pred_residual_type_probabilities": {
                        residual_types[type_idx]: round(float(type_probs[idx, type_idx]), 6)
                        for type_idx in range(len(residual_types))
                    },
                    "gold_novelty_label": batch["gold_novelty_label"][idx],
                    "pred_novelty_label": NOVELTY_LABELS[int(pred_final[idx])],
                    "ordinal_pred_novelty_label": NOVELTY_LABELS[int(pred_ordinal[idx])],
                    "direct_aux_pred_novelty_label": NOVELTY_LABELS[int(pred_direct_aux[idx])],
                    "residual_branch_pred_novelty_label": NOVELTY_LABELS[int(pred_residual_branch[idx])],
                    "rule_pred_novelty_label": NOVELTY_LABELS[int(pred_rule[idx])],
                    "pred_novelty_final_probabilities": {
                        NOVELTY_LABELS[label_idx]: round(float(novelty_probs[idx, label_idx]), 6)
                        for label_idx in range(len(NOVELTY_LABELS))
                    },
                    "pred_novelty_ordinal_probabilities": {
                        NOVELTY_LABELS[label_idx]: round(float(ordinal_probs[idx, label_idx]), 6)
                        for label_idx in range(len(NOVELTY_LABELS))
                    },
                    "pred_novelty_direct_probabilities": {
                        NOVELTY_LABELS[label_idx]: round(float(direct_probs[idx, label_idx]), 6)
                        for label_idx in range(len(NOVELTY_LABELS))
                    },
                    "pred_novelty_residual_probabilities": {
                        NOVELTY_LABELS[label_idx]: round(float(residual_novelty_probs[idx, label_idx]), 6)
                        for label_idx in range(len(NOVELTY_LABELS))
                    },
                    "pred_novelty_fusion_gates": {
                        "direct": round(float(fusion_gates[idx, 0]), 6),
                        "ordinal": round(float(fusion_gates[idx, 1]), 6),
                        "residual": round(float(fusion_gates[idx, 2]), 6),
                    },
                    "novelty_fusion_weights_used": {
                        "direct": round(float(novelty_fusion_weights[0]), 6),
                        "ordinal": round(float(novelty_fusion_weights[1]), 6),
                        "residual": round(float(novelty_fusion_weights[2]), 6),
                    },
                    "pred_moderate_high_probability": round(float(mh_prob[idx]), 6),
                    "pred_high_joint_score": round(float(high_joint_score[idx]), 6),
                    "moderate_high_threshold_used": round(float(moderate_high_threshold), 6),
                    "pred_combination_probability": round(float(combo_prob[idx]), 6),
                    "pred_combination_joint_score": round(float(combination_joint_score[idx]), 6),
                    "substantive_threshold_used": round(float(substantive_threshold), 6),
                    "combination_threshold_used": round(float(combination_threshold), 6),
                    "pred_coarse_residual_type": COARSE_TYPES[int(pred_coarse[idx])],
                    "pred_substantive_type_probability": round(float(substantive_prob[idx]), 6),
                    "pred_innovation_family": INNOVATION_FAMILIES[int(pred_family[idx])],
                    "semantic_unit_coverage_scores": [
                        round(float(value), 6) for value in coverage_scores[idx].tolist()
                    ],
                    "semantic_unit_residual_scores": [
                        round(float(value), 6) for value in residual_scores[idx].tolist()
                    ],
                    "semantic_unit_active_probabilities": [
                        round(float(value), 6) for value in unit_active[idx].tolist()
                    ],
                    "semantic_slot_masses": [
                        round(float(value), 6)
                        for value in semantic_slot_mass[idx].tolist()
                    ],
                    "semantic_unit_top_tokens": unit_top_tokens,
                    "residual_units": residual_units,
                    "covered_units": covered_units,
                    "interaction_residual_scores": [
                        round(float(value), 6) for value in interaction_scores[idx].tolist()
                    ],
                    "supporting_priors": [
                        {
                            **meta,
                            "learned_pair_weight": round(float(pair_weights[idx, meta_idx]), 6),
                        }
                        for meta_idx, meta in enumerate(batch["prior_meta"][idx])
                    ],
                }
            )
    y_type_arr = np.asarray(y_type, dtype=np.int64)
    p_type_arr = np.asarray(p_type, dtype=np.int64)
    y_novelty_arr = np.asarray(y_novelty, dtype=np.int64)
    type_prob_arr = np.concatenate(type_probability_batches, axis=0) if type_probability_batches else np.zeros((0, len(residual_types)))
    substantive_prob_arr = np.concatenate(substantive_probability_batches, axis=0) if substantive_probability_batches else np.zeros((0,))
    family_prob_arr = np.concatenate(family_probability_batches, axis=0) if family_probability_batches else np.zeros((0, len(INNOVATION_FAMILIES)))
    combination_prob_arr = np.concatenate(combination_probability_batches, axis=0) if combination_probability_batches else np.zeros((0,))
    direct_prob_arr = np.concatenate(direct_probability_batches, axis=0) if direct_probability_batches else np.zeros((0, len(NOVELTY_LABELS)))
    ordinal_prob_arr = np.concatenate(ordinal_probability_batches, axis=0) if ordinal_probability_batches else np.zeros((0, len(NOVELTY_LABELS)))
    residual_novelty_prob_arr = (
        np.concatenate(residual_novelty_probability_batches, axis=0)
        if residual_novelty_probability_batches
        else np.zeros((0, len(NOVELTY_LABELS)))
    )
    moderate_high_prob_arr = (
        np.concatenate(moderate_high_probability_batches, axis=0)
        if moderate_high_probability_batches
        else np.zeros((0,))
    )
    fusion_gate_arr = np.concatenate(fusion_gate_batches, axis=0) if fusion_gate_batches else np.zeros((0, 3))
    gate_summary: dict[str, Any] = {
        "mean_direct_gate": round(float(fusion_gate_arr[:, 0].mean()), 6) if len(fusion_gate_arr) else 0.0,
        "mean_ordinal_gate": round(float(fusion_gate_arr[:, 1].mean()), 6) if len(fusion_gate_arr) else 0.0,
        "mean_residual_gate": round(float(fusion_gate_arr[:, 2].mean()), 6) if len(fusion_gate_arr) else 0.0,
        "gate_collapse": bool(len(fusion_gate_arr) and float(fusion_gate_arr.mean(axis=0).max()) > 0.80),
        "by_gold_class": {},
    }
    for label_idx, label in enumerate(NOVELTY_LABELS):
        mask = y_novelty_arr == label_idx
        if bool(mask.any()) and len(fusion_gate_arr):
            gate_summary["by_gold_class"][label] = {
                "direct": round(float(fusion_gate_arr[mask, 0].mean()), 6),
                "ordinal": round(float(fusion_gate_arr[mask, 1].mean()), 6),
                "residual": round(float(fusion_gate_arr[mask, 2].mean()), 6),
            }
    residual_routing_calibration = calibrate_residual_routing(
        y_type_arr,
        type_prob_arr,
        substantive_prob_arr,
        family_prob_arr,
        combination_prob_arr,
        residual_types,
    ) if len(y_type_arr) else {}
    novelty_fusion_calibration = calibrate_novelty_fusion(
        y_novelty_arr,
        direct_prob_arr,
        ordinal_prob_arr,
        residual_novelty_prob_arr,
        moderate_high_prob_arr,
    ) if len(y_novelty_arr) else {}

    metrics = {
        "substantive_delta_score": regression_metrics(np.asarray(y_sub), np.asarray(p_sub)),
        "surface_change_score": regression_metrics(np.asarray(y_surface), np.asarray(p_surface)),
        "evidence_sufficiency": regression_metrics(np.asarray(y_evidence), np.asarray(p_evidence)),
        "estimated_unit_coverage_vs_gold_joint": regression_metrics(np.asarray(y_joint_gold), np.asarray(p_joint_est)),
        "estimated_unit_coverage_vs_predicted_joint": regression_metrics(np.asarray(y_joint_input), np.asarray(p_joint_est)),
        "estimated_unit_coverage_vs_joint_input": regression_metrics(np.asarray(y_joint_input), np.asarray(p_joint_est)),
        "residual_type": classification_metrics(y_type_arr, p_type_arr, residual_types),
        "fine_residual_type_by_coarse": fine_type_metrics_by_coarse(
            np.asarray(y_type),
            np.asarray(p_type),
            residual_types,
            model.config.fine_type_groups,
        ),
        "coarse_residual_type": classification_metrics(np.asarray(y_coarse), np.asarray(p_coarse), COARSE_TYPES),
        "minor_vs_substantive": binary_metrics(
            np.asarray(y_substantive_type),
            np.asarray(p_substantive_type),
            np.asarray(s_substantive_type),
        ),
        "minor_vs_substantive_classes": classification_metrics(
            np.asarray(y_substantive_type),
            np.asarray(p_substantive_type),
            ["minor_variant", "substantive_innovation"],
        ),
        "minor_vs_substantive_calibrated": calibrated_binary_metrics(
            np.asarray(y_substantive_type),
            np.asarray(s_substantive_type),
            threshold_min=0.20,
            threshold_max=0.80,
        ),
        "innovation_family": classification_metrics(
            np.asarray(y_innovation_family),
            np.asarray(p_innovation_family),
            INNOVATION_FAMILIES,
        ),
        "combination_binary": binary_metrics(
            np.asarray(y_combination), np.asarray(p_combination), np.asarray(s_combination)
        ),
        "combination_binary_calibrated": calibrated_binary_metrics(
            np.asarray(y_combination),
            np.asarray(s_combination),
            threshold_min=0.02,
            threshold_max=0.25,
        ),
        "combination_binary_raw": binary_metrics(
            np.asarray(y_combination), np.asarray(p_combination_raw), np.asarray(s_combination_raw)
        ),
        "combination_binary_raw_calibrated": calibrated_binary_metrics(
            np.asarray(y_combination),
            np.asarray(s_combination_raw),
            threshold_min=0.10,
            threshold_max=0.50,
        ),
        "novelty_direct": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_direct_aux), NOVELTY_LABELS, ordinal=True
        ),
        "novelty_direct_aux": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_direct_aux), NOVELTY_LABELS, ordinal=True
        ),
        "novelty_ordinal": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_ordinal), NOVELTY_LABELS, ordinal=True
        ),
        "novelty_residual_branch": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_residual_branch), NOVELTY_LABELS, ordinal=True
        ),
        "novelty_final": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_final), NOVELTY_LABELS, ordinal=True
        ),
        "moderate_high_binary": binary_metrics(
            np.asarray(y_moderate_high), np.asarray(p_moderate_high), np.asarray(s_moderate_high)
        ),
        "moderate_high_binary_calibrated": calibrated_binary_metrics(
            np.asarray(y_moderate_high),
            np.asarray(s_moderate_high),
            threshold_min=0.10,
            threshold_max=0.60,
        ),
        "novelty_rule_from_residual": classification_metrics(
            np.asarray(y_novelty), np.asarray(p_novelty_rule), NOVELTY_LABELS, ordinal=True
        ),
        "raw_novelty_score": {
            **regression_metrics(np.asarray(y_novelty, dtype=np.float32) / max(1, len(NOVELTY_LABELS) - 1), np.asarray(raw_novelty_scores)),
            "by_gold_label": score_quantiles_by_label(
                np.asarray(y_novelty),
                np.asarray(raw_novelty_scores),
                NOVELTY_LABELS,
            ),
            "note": "scored against label_id/3, a four-level staircase, not the continuous score",
        },
        "novelty_score_head": {
            **regression_metrics(np.asarray(y_novelty_score), np.asarray(p_novelty_score)),
            "by_gold_label": score_quantiles_by_label(
                np.asarray(y_novelty), np.asarray(p_novelty_score), NOVELTY_LABELS
            ),
        },
        "novelty_score_composed": {
            **regression_metrics(np.asarray(y_novelty_score), np.asarray(composed_novelty_scores)),
            "by_gold_label": score_quantiles_by_label(
                np.asarray(y_novelty), np.asarray(composed_novelty_scores), NOVELTY_LABELS
            ),
        },
        "novelty_score_gold_reference": {
            **regression_metrics(np.asarray(y_novelty_score), np.asarray(y_novelty_score)),
            "by_gold_label": score_quantiles_by_label(
                np.asarray(y_novelty), np.asarray(y_novelty_score), NOVELTY_LABELS
            ),
        },
        "novelty_from_score_head": classification_metrics(
            np.asarray(y_novelty),
            novelty_labels_from_score(
                np.asarray(p_novelty_score),
                np.asarray(p_evidence),
                novelty_score_cuts,
                evidence_gate=novelty_score_evidence_gate,
            ),
            NOVELTY_LABELS,
            ordinal=True,
        ),
        "novelty_score_evidence_gate_used": (
            None if novelty_score_evidence_gate is None else round(float(novelty_score_evidence_gate), 6)
        ),
        "novelty_from_score_head_no_gate": classification_metrics(
            np.asarray(y_novelty),
            novelty_labels_from_score(
                np.asarray(p_novelty_score), None, novelty_score_cuts, evidence_gate=None
            ),
            NOVELTY_LABELS,
            ordinal=True,
        ),
        "novelty_score_cuts_used": [round(float(c), 6) for c in novelty_score_cuts],
        "novelty_score_cut_calibration": calibrate_novelty_score_cuts(
            np.asarray(y_novelty), np.asarray(p_novelty_score), np.asarray(p_evidence)
        ),
        "semantic_units": semantic_metrics(
            semantic_attention_batches,
            coverage_batches,
            model.semantic_queries,
            active_batches,
            semantic_unit_batches,
            slot_assignment_batches,
            semantic_target_mask_batches,
        ),
        "residual_routing_calibration": residual_routing_calibration,
        "novelty_fusion_calibration": novelty_fusion_calibration,
        "novelty_fusion_gates": gate_summary,
    }
    metrics["residual_selection_score"] = residual_selection_score(metrics)
    residual_constraints_ok = residual_selection_constraints_ok(metrics)
    metrics["residual_selection_constraints_ok"] = residual_constraints_ok
    metrics["residual_selection_score_constrained"] = (
        metrics["residual_selection_score"] if residual_constraints_ok else float("-inf")
    )
    novelty_score_value, novelty_constraints_ok = novelty_selection_score(metrics)
    metrics["novelty_selection_score"] = novelty_score_value
    metrics["novelty_selection_constraints_ok"] = novelty_constraints_ok
    metrics["novelty_selection_score_constrained"] = (
        novelty_score_value if novelty_constraints_ok else float("-inf")
    )
    semantic_score_value = float(metrics["semantic_units"]["semantic_health_score"])
    semantic_constraints_ok = bool(metrics["semantic_units"]["semantic_constraints_ok"])
    semantic_selection_weight = float(
        np.clip(model.config.semantic_selection_weight, 0.0, 0.5)
    )
    explainable_novelty_score = round(
        (1.0 - semantic_selection_weight) * novelty_score_value
        + semantic_selection_weight * semantic_score_value,
        6,
    )
    metrics["semantic_selection_score"] = semantic_score_value
    metrics["semantic_selection_constraints_ok"] = semantic_constraints_ok
    metrics["novelty_explainable_selection_score"] = explainable_novelty_score
    metrics["novelty_explainable_selection_constraints_ok"] = bool(
        novelty_constraints_ok and semantic_constraints_ok
    )
    metrics["novelty_explainable_selection_score_constrained"] = (
        explainable_novelty_score
        if novelty_constraints_ok and semantic_constraints_ok
        else float("-inf")
    )
    metrics["total_selection_score"] = total_selection_score(metrics)
    metrics["total_selection_score_constrained"] = (
        metrics["total_selection_score"]
        if residual_constraints_ok and novelty_constraints_ok
        else float("-inf")
    )
    metrics["substantive_threshold_used"] = round(float(substantive_threshold), 6)
    metrics["combination_threshold_used"] = round(float(combination_threshold), 6)
    metrics["novelty_fusion_weights_used"] = {
        "direct": round(float(novelty_fusion_weights[0]), 6),
        "ordinal": round(float(novelty_fusion_weights[1]), 6),
        "residual": round(float(novelty_fusion_weights[2]), 6),
    }
    metrics["moderate_high_threshold_used"] = round(float(moderate_high_threshold), 6)
    return metrics, rows


def metric_value(metrics: dict[str, Any], selection_metric: str) -> float:
    current: Any = metrics
    for part in selection_metric.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Unknown selection metric path: {selection_metric}")
    return float(current)


def load_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.attn_implementation and args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.torch_dtype and args.torch_dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            major = int(str(transformers.__version__).split(".", 1)[0])
        except (AttributeError, ValueError):
            major = 5
        kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype_map[args.torch_dtype]
    return kwargs


def sample_rows(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:max_samples]


def build_residual_types(novelty_index: dict[str, dict[str, Any]], mode: str) -> list[str]:
    observed = {
        str(row.get("residual_type"))
        for row in novelty_index.values()
        if row.get("residual_type")
    }
    if mode == "observed":
        return [label for label in DEFAULT_RESIDUAL_TYPES if label in observed] + sorted(observed - set(DEFAULT_RESIDUAL_TYPES))
    return [label for label in DEFAULT_RESIDUAL_TYPES if label in observed or mode == "all"]


def build_fine_type_groups(residual_types: list[str]) -> dict[str, list[str]]:
    residual_set = set(residual_types)
    groups: dict[str, list[str]] = {}
    for coarse in COARSE_TYPES:
        labels = [label for label in FINE_TYPE_GROUPS[coarse] if label in residual_set]
        if not labels:
            labels = [residual_types[0]]
        groups[coarse] = labels
    return groups


def optimizer_groups(model: EGRDResidualPredictor, args: argparse.Namespace) -> list[dict[str, Any]]:
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return [
        {"params": encoder_params, "lr": args.lr * args.encoder_lr_ratio, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ]


def named_gradient_norm(model: EGRDResidualPredictor, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None or not name.startswith(prefixes):
            continue
        value = float(param.grad.detach().float().norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def gradient_norm_breakdown(model: EGRDResidualPredictor) -> dict[str, float]:
    return {
        "encoder_gradient_norm": named_gradient_norm(model, ("encoder.",)),
        "semantic_unit_gradient_norm": named_gradient_norm(
            model,
            (
                "semantic_queries",
                "support_mlp.",
                "coverage_pool_score.",
                "unit_active_head.",
                "unit_alignment_projection.",
                "prior_alignment_projection.",
                "coverage_gate.",
            ),
        ),
        "interaction_gradient_norm": named_gradient_norm(
            model,
            (
                "interaction_mlp.",
                "interaction_strength_head.",
                "interaction_pool_score.",
                "residual_fusion.",
            ),
        ),
        "type_head_gradient_norm": named_gradient_norm(
            model,
            (
                "minor_substantive_head.",
                "innovation_family_head.",
                "fine_type_heads.",
                "flat_type_head.",
                "combination_head.",
            ),
        ),
        "novelty_head_gradient_norm": named_gradient_norm(
            model,
            (
                "novelty_direct_head.",
                "novelty_ordinal_head.",
                "residual_novelty_head.",
                "novelty_fusion_gate.",
                "moderate_high_head.",
            ),
        ),
    }


def prediction_settings_from_metrics(args: argparse.Namespace, metrics: dict[str, Any]) -> dict[str, Any]:
    substantive_threshold = float(args.substantive_threshold)
    combination_threshold = float(args.combination_threshold)
    if args.calibrate_residual_routing:
        routing = metrics.get("residual_routing_calibration", {})
        substantive_threshold = float(routing.get("substantive_threshold", substantive_threshold))
        combination_threshold = float(routing.get("combination_threshold", combination_threshold))

    novelty_fusion_weights = (
        float(args.novelty_direct_weight),
        float(args.novelty_ordinal_weight),
        float(args.novelty_residual_weight),
    )
    moderate_high_threshold = float(args.moderate_high_threshold)
    if args.calibrate_novelty_fusion:
        novelty = metrics.get("novelty_fusion_calibration", {})
        novelty_fusion_weights = (
            float(novelty.get("direct_weight", novelty_fusion_weights[0])),
            float(novelty.get("ordinal_weight", novelty_fusion_weights[1])),
            float(novelty.get("residual_weight", novelty_fusion_weights[2])),
        )
        moderate_high_threshold = float(novelty.get("moderate_high_threshold", moderate_high_threshold))

    novelty_score_cuts = list(NOVELTY_FIXED_CUTS)
    novelty_score_evidence_gate: float | None = NOVELTY_EVIDENCE_GATE
    if args.calibrate_novelty_score_cuts:
        calibration = metrics.get("novelty_score_cut_calibration", {})
        fitted = calibration.get("cuts")
        if calibration.get("fitted") and isinstance(fitted, list) and len(fitted) == 3:
            novelty_score_cuts = [float(c) for c in fitted]
            gate = calibration.get("evidence_gate")
            novelty_score_evidence_gate = float(gate) if isinstance(gate, (int, float)) else None

    return {
        "substantive_threshold": substantive_threshold,
        "combination_threshold": combination_threshold,
        "novelty_fusion_weights": novelty_fusion_weights,
        "moderate_high_threshold": moderate_high_threshold,
        "novelty_score_cuts": novelty_score_cuts,
        "novelty_score_evidence_gate": novelty_score_evidence_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-dir", type=Path, default=None)
    parser.add_argument("--ranking-file", type=Path, default=Path("dataset/prior_ranking_dataset.jsonl"))
    parser.add_argument("--train-ranking-file", type=Path, default=None)
    parser.add_argument("--dev-ranking-file", type=Path, default=None)
    parser.add_argument("--test-ranking-file", type=Path, default=None)
    parser.add_argument("--novelty-file", type=Path, default=Path("dataset/idea_novelty_dataset.jsonl"))
    parser.add_argument("--joint-pred-dir", type=Path, default=None)
    parser.add_argument("--pair-text-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/stage_residual_egrd"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--init-from-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    parser.add_argument("--reinit-v22-modules", dest="reinit_v22_modules", action="store_true")
    parser.add_argument("--reinit-semantic-modules", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-semantic-units", type=int, default=8)
    parser.add_argument("--residual-type-mode", choices=["observed", "all"], default="observed")
    parser.add_argument(
        "--vocab-from-config",
        type=Path,
        default=None,
        help="Take residual_types and the three facet vocabularies from a training run's "
        "run_config.json instead of inducing them from the data. Required when applying a "
        "checkpoint to a corpus that lacks those annotations, otherwise the induced "
        "vocabularies differ in size and load_state_dict fails. Overrides --residual-type-mode.",
    )
    parser.add_argument("--use-gold-joint-evidence", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--encoder-lr-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--metadata-dim", type=int, default=32)
    parser.add_argument(
        "--facet-ablation",
        choices=["none", "unknown"],
        default="none",
        help="unknown = send every target's primary_aspect / contribution_type / task_family to the "
        "<unk> embedding row, measuring what the model loses on a corpus that carries no facet "
        "annotations. Note that row is never trained: our 5,225 training rows all have all three "
        "facets, so <unk> (index 0) receives no gradient and stays at its random init.",
    )
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--freeze-regression-heads-epochs", type=int, default=0)
    parser.add_argument("--unfreeze-last-layers", type=int, default=4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "float16", "bf16", "bfloat16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], default="auto")
    parser.add_argument("--slot-competition", dest="slot_competition", action="store_true", default=True)
    parser.add_argument("--no-slot-competition", dest="slot_competition", action="store_false")
    parser.add_argument(
        "--unit-interaction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ablation B4: --no-unit-interaction zeroes the pairwise unit-interaction branch, so "
        "nontrivial_combination has to be read off single units. Note --num-semantic-units 1 "
        "already disables it implicitly, which is why B1 and B4 are confounded.",
    )
    parser.add_argument(
        "--ablate-feature",
        action="append",
        default=[],
        choices=list(ABLATION_TARGETS),
        help="Zero one input channel on entry, keeping shapes and parameter count identical. "
        "Repeatable. joint = the joint stage's score and posterior; pair = the pair stage's "
        "scores and per-pair features; pair_meta = the same pair label/score where they are "
        "embedded in the prompt text (cut it together with pair, or the signal survives); "
        "scalars = the D/S/E heads fed into novelty_features; "
        "target = the target text representation; egrd = the residual and covered "
        "representations, i.e. the whole decomposition block's output, with egrd_residual "
        "and egrd_covered cutting one half each; global = the coverage "
        "feature vector; type = the residual-type posterior.",
    )
    parser.add_argument(
        "--residual-direction",
        choices=["residual", "covered"],
        default="residual",
        help="Ablation B5: the weighting of unit_residual_representation. residual = "
        "active x (1 - coverage) is the method's decomposition; covered = active x coverage "
        "flips the subtraction, keeping shapes and parameter count identical.",
    )
    parser.add_argument(
        "--semantic-assignment-mode",
        choices=["legacy", "softmax", "sparsemax"],
        default="sparsemax",
    )
    parser.add_argument("--semantic-score-temperature", type=float, default=0.07)
    parser.add_argument("--semantic-score-smoothing-kernel", type=int, default=3)
    parser.add_argument("--semantic-sparsemax-weight", type=float, default=0.95)
    parser.add_argument("--semantic-balance-iterations", type=int, default=3)
    parser.add_argument("--semantic-min-slot-mass", type=float, default=0.25)
    parser.add_argument("--semantic-selection-weight", type=float, default=0.12)
    parser.add_argument("--coverage-feature-noise-std", type=float, default=0.03)
    parser.add_argument("--coverage-feature-dropout", type=float, default=0.10)
    parser.add_argument("--flat-type-blend-weight", type=float, default=0.15)
    parser.add_argument("--novelty-fusion-mode", choices=["fixed", "dynamic"], default="fixed")
    parser.add_argument("--novelty-direct-weight", type=float, default=0.50)
    parser.add_argument("--novelty-ordinal-weight", type=float, default=0.35)
    parser.add_argument("--novelty-residual-weight", type=float, default=0.15)
    parser.add_argument("--gold-coverage-input-ratio-start", type=float, default=0.5)
    parser.add_argument("--gold-coverage-input-ratio-mid", type=float, default=0.2)
    parser.add_argument("--gold-coverage-input-ratio-end", type=float, default=0.0)
    parser.add_argument("--gold-coverage-mix-stage-epochs", type=int, default=1)
    parser.add_argument("--huber-beta", type=float, default=0.08)
    parser.add_argument("--substantive-loss-weight", type=float, default=1.0)
    parser.add_argument("--surface-loss-weight", type=float, default=0.6)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.8)
    parser.add_argument("--regression-ranking-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--evidence-ranking-weight",
        type=float,
        default=0.5,
        help="Multiplier on the evidence head's ranking term, relative to the substantive head. "
        "The v2.3 value of 0.5 leaves evidence the weakest-ranked head and it collapses onto "
        "the median of its target; raise it to recover moderate/high separation.",
    )
    parser.add_argument("--surface-ranking-weight", type=float, default=0.5)
    parser.add_argument("--coverage-loss-weight", type=float, default=0.3)
    parser.add_argument("--gold-coverage-loss-weight", type=float, default=1.0)
    parser.add_argument("--predicted-coverage-consistency-loss-weight", type=float, default=0.2)
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--minor-substantive-loss-weight", type=float, default=0.3)
    parser.add_argument("--innovation-family-loss-weight", type=float, default=0.3)
    parser.add_argument("--conditional-fine-loss-weight", type=float, default=0.3)
    parser.add_argument("--flat-type-loss-weight", type=float, default=0.1)
    parser.add_argument("--combination-binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--combination-consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--combination-family-consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--combination-pos-weight", type=float, default=0.0)
    parser.add_argument("--substantive-threshold", type=float, default=0.50)
    parser.add_argument("--combination-threshold", type=float, default=0.08)
    parser.add_argument(
        "--calibrate-residual-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--novelty-loss-weight", type=float, default=0.8)
    parser.add_argument(
        "--novelty-score-loss-weight",
        type=float,
        default=0.6,
        help="Weight of the continuous novelty-score head. Set to 0 to reproduce the v2.3 heads.",
    )
    parser.add_argument(
        "--novelty-score-rank-weight",
        type=float,
        default=1.0,
        help="Weight of the pairwise ranking term inside the novelty-score loss. "
        "Thresholding only depends on order, so this drives the metric that matters.",
    )
    parser.add_argument(
        "--calibrate-novelty-score-cuts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit the score head's three cut points on dev, then freeze them for test.",
    )
    parser.add_argument("--ordinal-novelty-loss-weight", type=float, default=0.5)
    parser.add_argument("--direct-novelty-loss-weight", type=float, default=0.5)
    parser.add_argument("--residual-novelty-loss-weight", type=float, default=0.4)
    parser.add_argument("--final-novelty-loss-weight", type=float, default=0.6)
    parser.add_argument("--moderate-high-loss-weight", type=float, default=0.6)
    parser.add_argument("--moderate-high-pos-weight", type=float, default=0.0)
    parser.add_argument("--high-margin-loss-weight", type=float, default=0.2)
    parser.add_argument("--high-margin", type=float, default=0.10)
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.1)
    parser.add_argument("--gate-balance-loss-weight", type=float, default=0.02)
    parser.add_argument("--gate-balance-target", type=float, nargs=3, default=[0.45, 0.35, 0.20])
    parser.add_argument("--moderate-high-threshold", type=float, default=0.2688)
    parser.add_argument(
        "--calibrate-novelty-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semantic-diversity-loss-weight", type=float, default=0.10)
    parser.add_argument("--attention-overlap-loss-weight", type=float, default=0.35)
    parser.add_argument("--query-diversity-loss-weight", type=float, default=0.05)
    parser.add_argument("--unit-usage-balance-loss-weight", type=float, default=0.10)
    parser.add_argument("--slot-entropy-loss-weight", type=float, default=0.30)
    parser.add_argument("--semantic-unit-separation-loss-weight", type=float, default=0.20)
    parser.add_argument("--max-semantic-unit-cosine", type=float, default=0.80)
    parser.add_argument("--unit-collapse-loss-weight", type=float, default=0.05)
    parser.add_argument("--min-unit-coverage-variance", type=float, default=0.005)
    parser.add_argument("--hard-negative-high-probability", type=float, default=0.55)
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    parser.add_argument("--semantic-top-token-count", type=int, default=6)
    parser.add_argument(
        "--dump-embeddings",
        action="store_true",
        help="With --eval-only-checkpoint, also write per-split stage representations to "
             "<split>_embeddings.npz for offline analysis.",
    )
    parser.add_argument("--surface-penalty-lambda", type=float, default=0.45)
    parser.add_argument("--selection-metric", default="novelty_selection_score_constrained")
    parser.add_argument(
        "--final-checkpoint",
        choices=["residual", "novelty", "novelty_explainable", "total"],
        default="novelty",
    )
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    novelty_index = load_novelty_index(args.novelty_file)
    split_rankings, split_source = load_ranking_splits(args, novelty_index)
    split_rankings["train"] = sample_rows(split_rankings["train"], args.max_train_samples, args.seed)
    if args.max_eval_samples > 0:
        split_rankings["dev"] = sample_rows(split_rankings["dev"], args.max_eval_samples, args.seed + 1)
        split_rankings["test"] = sample_rows(split_rankings["test"], args.max_eval_samples, args.seed + 2)

    joint_predictions = load_joint_predictions(args.joint_pred_dir)
    text_lookup = build_text_lookup(novelty_index, args.pair_text_file)
    residual_types = build_residual_types(novelty_index, args.residual_type_mode)
    residual_to_id = {label: idx for idx, label in enumerate(residual_types)}
    fine_type_groups = build_fine_type_groups(residual_types)
    if args.vocab_from_config is not None:
        source = json.loads(args.vocab_from_config.read_text(encoding="utf-8"))
        source = source.get("model_config", source)
        residual_types = list(source["residual_types"])
        residual_to_id = {label: idx for idx, label in enumerate(residual_types)}
        fine_type_groups = build_fine_type_groups(residual_types)
        aspect_vocab = dict(source["aspect_vocab"])
        contribution_vocab = dict(source["contribution_vocab"])
        task_vocab = dict(source["task_vocab"])
        print(
            f"vocab from {args.vocab_from_config}: residual_types={len(residual_types)} "
            f"aspect={len(aspect_vocab)} contribution={len(contribution_vocab)} task={len(task_vocab)}"
        )
    else:
        aspect_vocab = build_vocab([row.get("primary_aspect") for row in novelty_index.values()])
        contribution_vocab = build_vocab([row.get("contribution_type") for row in novelty_index.values()])
        task_vocab = build_vocab([row.get("task_family") for row in novelty_index.values()])

    model_config = EGRDConfig(
        model_name=args.model_name,
        top_k=args.top_k,
        max_length=args.max_length,
        num_semantic_units=args.num_semantic_units,
        residual_types=residual_types,
        fine_type_groups=fine_type_groups,
        slot_competition=args.slot_competition,
        unit_interaction=args.unit_interaction,
        ablate_features=tuple(sorted(set(args.ablate_feature))),
        residual_direction=args.residual_direction,
        semantic_assignment_mode=args.semantic_assignment_mode,
        semantic_score_temperature=args.semantic_score_temperature,
        semantic_score_smoothing_kernel=args.semantic_score_smoothing_kernel,
        semantic_sparsemax_weight=args.semantic_sparsemax_weight,
        semantic_balance_iterations=args.semantic_balance_iterations,
        semantic_min_slot_mass=args.semantic_min_slot_mass,
        semantic_selection_weight=args.semantic_selection_weight,
        coverage_feature_noise_std=args.coverage_feature_noise_std,
        coverage_feature_dropout=args.coverage_feature_dropout,
        flat_type_blend_weight=args.flat_type_blend_weight,
        novelty_fusion_mode=args.novelty_fusion_mode,
        novelty_direct_weight=args.novelty_direct_weight,
        novelty_ordinal_weight=args.novelty_ordinal_weight,
        novelty_residual_weight=args.novelty_residual_weight,
        aspect_vocab=aspect_vocab,
        contribution_vocab=contribution_vocab,
        task_vocab=task_vocab,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    datasets = {
        split: ResidualEGRDDataset(
            rows,
            novelty_index,
            joint_predictions,
            text_lookup,
            model_config,
            aspect_vocab,
            contribution_vocab,
            task_vocab,
            residual_to_id,
            use_gold_joint_evidence=args.use_gold_joint_evidence,
            facet_ablation=args.facet_ablation,
        )
        for split, rows in split_rankings.items()
    }
    collator = EGRDCollator(tokenizer, args.max_length, args.top_k)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loaders = {
        split: DataLoader(datasets[split], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
        for split in ["train", "dev", "test"]
    }
    loss_context = build_loss_context(datasets["train"], model_config)
    if args.combination_pos_weight > 0:
        loss_context["combination_pos_weight"] = float(min(5.0, args.combination_pos_weight))
    if args.moderate_high_pos_weight > 0:
        loss_context["moderate_high_pos_weight"] = float(args.moderate_high_pos_weight)

    device = device_from_arg(args.device)
    model = EGRDResidualPredictor(
        model_config,
        dropout=args.dropout,
        metadata_dim=args.metadata_dim,
        load_kwargs=load_kwargs_from_args(args),
    )
    encoder_checkpoint_info = None
    init_checkpoint_info = None
    if args.init_from_checkpoint is not None:
        if args.reinit_v22_modules:
            checkpoint_skip_prefixes = V22_REINIT_PREFIXES
        elif args.reinit_semantic_modules:
            checkpoint_skip_prefixes = V24_SEMANTIC_REINIT_PREFIXES
        else:
            checkpoint_skip_prefixes = ()
        init_checkpoint_info = load_matching_checkpoint(
            model,
            args.init_from_checkpoint,
            skip_prefixes=checkpoint_skip_prefixes,
        )
    elif args.encoder_checkpoint is not None:
        encoder_checkpoint_info = load_encoder_checkpoint(model, args.encoder_checkpoint)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        try:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.encoder.gradient_checkpointing_enable()
            if hasattr(model.encoder, "enable_input_require_grads"):
                model.encoder.enable_input_require_grads()
    freeze_encoder(model, 0)
    model.float().to(device)

    optimizer = torch.optim.AdamW(optimizer_groups(model, args))
    amp_dtype = amp_dtype_from_arg(device, args.amp_dtype)
    scaler_enabled = bool(args.amp and device.type == "cuda" and amp_dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    run_config = {
        **{key: jsonable(value) for key, value in vars(args).items()},
        "ranking_dir": str(args.ranking_dir) if args.ranking_dir else None,
        "ranking_file": str(args.ranking_file) if args.ranking_file else None,
        "train_ranking_file": str(args.train_ranking_file) if args.train_ranking_file else None,
        "dev_ranking_file": str(args.dev_ranking_file) if args.dev_ranking_file else None,
        "test_ranking_file": str(args.test_ranking_file) if args.test_ranking_file else None,
        "novelty_file": str(args.novelty_file),
        "joint_pred_dir": str(args.joint_pred_dir) if args.joint_pred_dir else None,
        "pair_text_file": [str(path) for path in args.pair_text_file],
        "split_source": split_source,
        "labels": {
            "pair": PAIR_LABELS,
            "joint": JOINT_LABELS,
            "residual_types": residual_types,
            "coarse_residual_types": COARSE_TYPES,
            "fine_type_groups": fine_type_groups,
            "novelty": NOVELTY_LABELS,
        },
        "feature_names": {
            "pair": PAIR_FEATURE_NAMES,
            "global": GLOBAL_FEATURE_NAMES,
        },
        "model_config": asdict(model_config),
        "encoder_checkpoint": encoder_checkpoint_info,
        "init_checkpoint": init_checkpoint_info,
        "loss_context": loss_context,
        "effective_amp_dtype": str(amp_dtype).replace("torch.", ""),
        "note": (
            "Gold residual labels supervise losses only. "
            "Gold joint coverage enters model inputs only when --use-gold-joint-evidence is set."
        ),
    }
    write_json(args.output_dir / "run_config.json", run_config)

    if args.eval_only_checkpoint is not None:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        dev_probe_metrics, _ = predict(
            model,
            eval_loaders["dev"],
            device,
            residual_types,
            tokenizer,
            surface_penalty_lambda=args.surface_penalty_lambda,
            substantive_threshold=args.substantive_threshold,
            combination_threshold=args.combination_threshold,
            novelty_fusion_weights=(args.novelty_direct_weight, args.novelty_ordinal_weight, args.novelty_residual_weight),
            moderate_high_threshold=args.moderate_high_threshold,
            semantic_top_token_count=0,
        )
        eval_prediction_settings = prediction_settings_from_metrics(args, dev_probe_metrics)
        final_metrics: dict[str, Any] = {
            "task": "evidence_grounded_residual_decomposition_v2_4_eval_only",
            "eval_only_checkpoint": str(args.eval_only_checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
            "prediction_settings": eval_prediction_settings,
            "load_state": {
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            },
            "split_source": split_source,
            "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
            "labels": run_config["labels"],
        }
        all_prediction_rows = []
        for split in ["train", "dev", "test"]:
            sink: dict[str, list[Any]] | None = {} if args.dump_embeddings else None
            split_metrics, split_rows = predict(
                model,
                eval_loaders[split],
                device,
                residual_types,
                tokenizer,
                surface_penalty_lambda=args.surface_penalty_lambda,
                semantic_top_token_count=args.semantic_top_token_count,
                embedding_sink=sink,
                **eval_prediction_settings,
            )
            if sink:
                ids = sink.pop("target_idea_id")
                np.savez_compressed(
                    args.output_dir / f"{split}_embeddings.npz",
                    target_idea_id=np.asarray(ids),
                    **{name: np.concatenate(chunks, axis=0) for name, chunks in sink.items()},
                )
            final_metrics[split] = split_metrics
            write_jsonl(args.output_dir / f"{split}_predictions.jsonl", split_rows)
            all_prediction_rows.extend(dict(row, split=split) for row in split_rows)
        write_jsonl(args.output_dir / "all_predictions.jsonl", all_prediction_rows)
        write_json(args.output_dir / "metrics.json", final_metrics)
        print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
        return 0

    best_custom_value = -float("inf")
    best_custom_epoch = -1
    best_residual_value = -float("inf")
    best_residual_epoch = -1
    best_novelty_value = -float("inf")
    best_novelty_epoch = -1
    best_novelty_explainable_value = -float("inf")
    best_novelty_explainable_epoch = -1
    best_novelty_explainable_constraints_ok = False
    best_total_value = -float("inf")
    best_total_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_encoder_epochs + 1:
            unfreeze_encoder(model, args.unfreeze_last_layers)
            trainable_encoder = sum(
                1 for name, param in model.named_parameters()
                if name.startswith("encoder.") and param.requires_grad
            )
            print(
                f"[epoch {epoch}] unfroze encoder: {trainable_encoder} trainable tensors",
                flush=True,
            )
            if trainable_encoder == 0:
                print(
                    "  WARNING: nothing became trainable -- check --unfreeze-last-layers",
                    flush=True,
                )
        set_regression_heads_trainable(model, epoch > args.freeze_regression_heads_epochs)
        model.train()
        loss_sums: dict[str, float] = defaultdict(float)
        steps = 0
        attempted_steps = 0
        nonfinite_loss_steps = 0
        nonfinite_gradient_steps = 0
        coverage_mix_ratio = gold_coverage_input_ratio(args, epoch)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            batch = maybe_mix_coverage_inputs(batch, args, epoch)
            attempted_steps += 1
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp, amp_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    target_token_mask=batch["target_token_mask"],
                    pair_mask=batch["pair_mask"],
                    pair_features=batch["pair_features"],
                    global_features=batch["global_features"],
                    aspect_id=batch["aspect_id"],
                    contribution_id=batch["contribution_id"],
                    task_id=batch["task_id"],
                    joint_normalized_entropy=batch["joint_normalized_entropy"],
                )
                loss, parts = compute_loss(outputs, batch, model, args, loss_context)
            if not bool(torch.isfinite(loss).all()):
                nonfinite_loss_steps += 1
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(loss="nonfinite")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_breakdown = gradient_norm_breakdown(model)
            if (
                epoch > args.freeze_encoder_epochs
                and steps == 0
                and grad_breakdown.get("encoder_gradient_norm", 0.0) == 0.0
            ):
                print(
                    "  WARNING: encoder is unfrozen but its gradient norm is 0. "
                    "Gradient checkpointing may be severing the graph again.",
                    flush=True,
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            grad_norm_value = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            if not math.isfinite(grad_norm_value):
                nonfinite_gradient_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", grad="nonfinite")
                continue
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for key, value in parts.items():
                loss_sums[key] += value
            for key, value in grad_breakdown.items():
                loss_sums[key] += value
            loss_sums["gradient_norm"] += grad_norm_value
            progress.set_postfix(loss=f"{loss_sums['loss'] / max(1, steps):.4f}")

        train_loss = {key: round(value / max(1, steps), 6) for key, value in loss_sums.items()}
        train_loss["attempted_steps"] = int(attempted_steps)
        train_loss["optimizer_steps"] = int(steps)
        train_loss["nonfinite_loss_steps"] = int(nonfinite_loss_steps)
        train_loss["nonfinite_gradient_steps"] = int(nonfinite_gradient_steps)
        train_loss["gold_coverage_input_ratio"] = round(float(coverage_mix_ratio), 6)
        dev_metrics, _ = predict(
            model,
            eval_loaders["dev"],
            device,
            residual_types,
            tokenizer,
            surface_penalty_lambda=args.surface_penalty_lambda,
            substantive_threshold=args.substantive_threshold,
            combination_threshold=args.combination_threshold,
            novelty_fusion_weights=(args.novelty_direct_weight, args.novelty_ordinal_weight, args.novelty_residual_weight),
            moderate_high_threshold=args.moderate_high_threshold,
            semantic_top_token_count=0,
        )
        prediction_settings = prediction_settings_from_metrics(args, dev_metrics)
        dev_metrics, _ = predict(
            model,
            eval_loaders["dev"],
            device,
            residual_types,
            tokenizer,
            surface_penalty_lambda=args.surface_penalty_lambda,
            semantic_top_token_count=0,
            **prediction_settings,
        )
        current = metric_value(dev_metrics, args.selection_metric)
        residual_current = float(dev_metrics["residual_selection_score_constrained"])
        novelty_current = float(dev_metrics["novelty_selection_score_constrained"])
        novelty_explainable_current = float(
            dev_metrics["novelty_explainable_selection_score_constrained"]
        )
        novelty_explainable_raw = float(
            dev_metrics["novelty_explainable_selection_score"]
        )
        novelty_explainable_constraints_ok = bool(
            dev_metrics["novelty_explainable_selection_constraints_ok"]
        )
        total_current = float(dev_metrics["total_selection_score_constrained"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev": dev_metrics,
            "selection_metric": args.selection_metric,
            "selection_value": current,
            "residual_selection_score": dev_metrics["residual_selection_score"],
            "residual_selection_score_constrained": residual_current,
            "residual_selection_constraints_ok": dev_metrics["residual_selection_constraints_ok"],
            "novelty_selection_score": dev_metrics["novelty_selection_score"],
            "novelty_selection_score_constrained": novelty_current,
            "novelty_selection_constraints_ok": dev_metrics["novelty_selection_constraints_ok"],
            "semantic_selection_score": dev_metrics["semantic_selection_score"],
            "semantic_selection_constraints_ok": dev_metrics["semantic_selection_constraints_ok"],
            "novelty_explainable_selection_score": dev_metrics[
                "novelty_explainable_selection_score"
            ],
            "novelty_explainable_selection_score_constrained": novelty_explainable_current,
            "novelty_explainable_selection_constraints_ok": dev_metrics[
                "novelty_explainable_selection_constraints_ok"
            ],
            "total_selection_score": dev_metrics["total_selection_score"],
            "total_selection_score_constrained": total_current,
            "prediction_settings": prediction_settings,
        }
        history.append(epoch_record)
        write_json(args.output_dir / f"dev_epoch_{epoch}.json", epoch_record)

        def save_checkpoint(best_dir: Path, metric_name: str, metric_value_for_save: float) -> None:
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(model_config),
                    "run_config": run_config,
                    "epoch": epoch,
                    "selection_metric": metric_name,
                    "selection_value": metric_value_for_save,
                    "prediction_settings": prediction_settings,
                },
                best_dir / "checkpoint.pt",
            )
            tokenizer.save_pretrained(best_dir / "tokenizer")
            write_json(best_dir / "metrics.json", epoch_record)

        if best_custom_epoch < 0 or current > best_custom_value:
            best_custom_value = current
            best_custom_epoch = epoch
            save_checkpoint(args.output_dir / "best", args.selection_metric, best_custom_value)
        if best_residual_epoch < 0 or residual_current > best_residual_value:
            best_residual_value = residual_current
            best_residual_epoch = epoch
            save_checkpoint(args.output_dir / "best_residual", "residual_selection_score_constrained", best_residual_value)
        if best_novelty_epoch < 0 or novelty_current > best_novelty_value:
            best_novelty_value = novelty_current
            best_novelty_epoch = epoch
            save_checkpoint(args.output_dir / "best_novelty", "novelty_selection_score_constrained", best_novelty_value)
        should_save_explainable = (
            best_novelty_explainable_epoch < 0
            or (
                novelty_explainable_constraints_ok
                and not best_novelty_explainable_constraints_ok
            )
            or (
                novelty_explainable_constraints_ok
                == best_novelty_explainable_constraints_ok
                and novelty_explainable_raw > best_novelty_explainable_value
            )
        )
        if should_save_explainable:
            best_novelty_explainable_value = novelty_explainable_raw
            best_novelty_explainable_epoch = epoch
            best_novelty_explainable_constraints_ok = (
                novelty_explainable_constraints_ok
            )
            save_checkpoint(
                args.output_dir / "best_novelty_explainable",
                (
                    "novelty_explainable_selection_score_constrained"
                    if novelty_explainable_constraints_ok
                    else "novelty_explainable_selection_score"
                ),
                best_novelty_explainable_value,
            )
        if best_total_epoch < 0 or total_current > best_total_value:
            best_total_value = total_current
            best_total_epoch = epoch
            save_checkpoint(args.output_dir / "best_total", "total_selection_score_constrained", best_total_value)

    final_dir = args.output_dir / {
        "novelty": "best_novelty",
        "novelty_explainable": "best_novelty_explainable",
        "residual": "best_residual",
        "total": "best_total",
    }[args.final_checkpoint]
    checkpoint = torch.load(final_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_prediction_settings = checkpoint.get(
        "prediction_settings",
        {
            "substantive_threshold": args.substantive_threshold,
            "combination_threshold": args.combination_threshold,
            "novelty_fusion_weights": (
                args.novelty_direct_weight,
                args.novelty_ordinal_weight,
                args.novelty_residual_weight,
            ),
            "moderate_high_threshold": args.moderate_high_threshold,
        },
    )
    final_metrics: dict[str, Any] = {
        "task": "evidence_grounded_residual_decomposition_v2_4",
        "final_checkpoint": args.final_checkpoint,
        "final_checkpoint_dir": str(final_dir),
        "prediction_settings": final_prediction_settings,
        "best_epoch": int(checkpoint["epoch"]),
        "best_selection_value": float(checkpoint["selection_value"]),
        "selection_metric": args.selection_metric,
        "best_custom": {
            "epoch": best_custom_epoch,
            "metric": args.selection_metric,
            "value": best_custom_value,
        },
        "best_residual": {
            "epoch": best_residual_epoch,
            "metric": "residual_selection_score_constrained",
            "value": best_residual_value,
        },
        "best_novelty": {
            "epoch": best_novelty_epoch,
            "metric": "novelty_selection_score_constrained",
            "value": best_novelty_value,
        },
        "best_novelty_explainable": {
            "epoch": best_novelty_explainable_epoch,
            "metric": (
                "novelty_explainable_selection_score_constrained"
                if best_novelty_explainable_constraints_ok
                else "novelty_explainable_selection_score"
            ),
            "value": best_novelty_explainable_value,
            "constraints_ok": best_novelty_explainable_constraints_ok,
        },
        "best_total": {
            "epoch": best_total_epoch,
            "metric": "total_selection_score_constrained",
            "value": best_total_value,
        },
        "split_source": split_source,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "labels": run_config["labels"],
        "history": history,
    }
    all_prediction_rows = []
    for split in ["train", "dev", "test"]:
        split_metrics, split_rows = predict(
            model,
            eval_loaders[split],
            device,
            residual_types,
            tokenizer,
            surface_penalty_lambda=args.surface_penalty_lambda,
            semantic_top_token_count=args.semantic_top_token_count,
            **final_prediction_settings,
        )
        final_metrics[split] = split_metrics
        write_jsonl(args.output_dir / f"{split}_predictions.jsonl", split_rows)
        all_prediction_rows.extend(dict(row, split=split) for row in split_rows)
    write_jsonl(args.output_dir / "all_predictions.jsonl", all_prediction_rows)
    write_json(args.output_dir / "metrics.json", final_metrics)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
