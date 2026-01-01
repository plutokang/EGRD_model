from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PAIR_LABELS = ["not_covering", "related_not_covering", "partial_cover", "large_cover"]
PAIR_TO_ID = {label: idx for idx, label in enumerate(PAIR_LABELS)}
COVERING_LABELS = {"partial_cover", "large_cover"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def label_id(label: Any) -> int:
    return PAIR_TO_ID.get(str(label or "not_covering"), 0)


def prediction_label(row: dict[str, Any]) -> str:
    return str(row.get("calibrated_pair_label") or row.get("pred_pair_label") or "not_covering")


def prediction_score(row: dict[str, Any]) -> float:
    if "calibrated_score" in row:
        return safe_float(row.get("calibrated_score"), 0.0)
    return safe_float(row.get("pred_coverage_score"), 0.0)


def rank_score(row: dict[str, Any], label_bonus: float) -> float:
    pred_label = prediction_label(row)
    pred_level = label_id(pred_label) / 3.0
    return prediction_score(row) + label_bonus * pred_level


def average_precision(relevant: list[int]) -> float:
    hits = 0
    total = 0.0
    for idx, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            total += hits / idx
    return total / max(1, sum(relevant))


def reciprocal_rank(relevant: list[int]) -> float:
    for idx, rel in enumerate(relevant, start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def dcg(relevance: list[float]) -> float:
    return sum((2**rel - 1.0) / np.log2(idx + 2) for idx, rel in enumerate(relevance))


def ndcg(relevance: list[float], k: int) -> float:
    rel = relevance[:k]
    best = sorted(relevance, reverse=True)[:k]
    denom = dcg(best)
    return dcg(rel) / denom if denom else 0.0


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split") or "unknown")
        target_id = str(row.get("target_idea_id") or "")
        if not target_id:
            continue
        groups[(split, target_id)].append(row)
    return groups


def candidate_output(row: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    gold_label = str(row.get("gold_pair_label") or row.get("coverage_label") or "not_covering")
    pred_label = prediction_label(row)
    return {
        "rank": rank,
        "prior_idea_id": row.get("prior_idea_id"),
        "rank_score": round(float(score), 6),
        "calibrated_score": round(prediction_score(row), 6),
        "calibrated_pair_label": pred_label,
        "raw_pred_pair_label": row.get("raw_pred_pair_label") or row.get("pred_pair_label"),
        "gold_pair_label": gold_label,
        "gold_coverage_score": safe_float(row.get("gold_coverage_score"), safe_float(row.get("coverage_score"), 0.0)),
        "is_gold_covering": gold_label in COVERING_LABELS,
        "is_pred_covering": pred_label in COVERING_LABELS,
        "pred_pair_probabilities": row.get("pred_pair_probabilities"),
    }


def build_rankings(rows: list[dict[str, Any]], top_k: int, label_bonus: float) -> dict[str, list[dict[str, Any]]]:
    grouped = group_rows(rows)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (split, target_id), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        scored = [(rank_score(row, label_bonus), row) for row in items]
        scored.sort(
            key=lambda item: (
                item[0],
                safe_float(item[1].get("pred_coverage_score"), 0.0),
                label_id(prediction_label(item[1])),
                str(item[1].get("prior_idea_id") or ""),
            ),
            reverse=True,
        )
        candidates = [
            candidate_output(row, rank=rank, score=score)
            for rank, (score, row) in enumerate(scored[:top_k], start=1)
        ]
        all_gold_covering = sum(str(row.get("gold_pair_label") or "") in COVERING_LABELS for row in items)
        out[split].append(
            {
                "split": split,
                "target_idea_id": target_id,
                "num_candidates": len(items),
                "num_gold_covering_candidates": int(all_gold_covering),
                "top_k": top_k,
                "ranked_priors": candidates,
            }
        )
    return out


def ranking_metrics(rankings: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, Any]:
    target_count = len(rankings)
    with_covering = [row for row in rankings if int(row.get("num_gold_covering_candidates") or 0) > 0]
    metrics: dict[str, Any] = {
        "targets": target_count,
        "targets_with_gold_covering": len(with_covering),
        "targets_without_gold_covering": target_count - len(with_covering),
    }
    for k in cutoffs:
        hit = []
        recall = []
        precision = []
        pred_covering_rate = []
        for row in with_covering:
            priors = row["ranked_priors"][:k]
            gold_total = max(1, int(row.get("num_gold_covering_candidates") or 0))
            gold_hits = sum(bool(prior.get("is_gold_covering")) for prior in priors)
            pred_covering = sum(bool(prior.get("is_pred_covering")) for prior in priors)
            hit.append(1.0 if gold_hits else 0.0)
            recall.append(gold_hits / gold_total)
            precision.append(gold_hits / max(1, len(priors)))
            pred_covering_rate.append(pred_covering / max(1, len(priors)))
        metrics[f"hit@{k}"] = round(float(np.mean(hit)), 4) if hit else 0.0
        metrics[f"recall@{k}"] = round(float(np.mean(recall)), 4) if recall else 0.0
        metrics[f"precision@{k}"] = round(float(np.mean(precision)), 4) if precision else 0.0
        metrics[f"pred_covering_rate@{k}"] = round(float(np.mean(pred_covering_rate)), 4) if pred_covering_rate else 0.0

    mrr = []
    map_scores = []
    ndcg_scores: dict[int, list[float]] = {k: [] for k in cutoffs}
    for row in with_covering:
        rel = [1 if prior.get("is_gold_covering") else 0 for prior in row["ranked_priors"]]
        graded = [safe_float(prior.get("gold_coverage_score"), 0.0) for prior in row["ranked_priors"]]
        mrr.append(reciprocal_rank(rel))
        map_scores.append(average_precision(rel))
        for k in cutoffs:
            ndcg_scores[k].append(ndcg(graded, k))
    metrics["mrr"] = round(float(np.mean(mrr)), 4) if mrr else 0.0
    metrics["map"] = round(float(np.mean(map_scores)), 4) if map_scores else 0.0
    for k in cutoffs:
        metrics[f"ndcg@{k}"] = round(float(np.mean(ndcg_scores[k])), 4) if ndcg_scores[k] else 0.0

    top1_labels = Counter()
    for row in rankings:
        priors = row.get("ranked_priors") or []
        if priors:
            top1_labels[str(priors[0].get("calibrated_pair_label") or "")] += 1
    metrics["top1_pred_label_distribution"] = dict(top1_labels)
    return metrics


def parse_cutoffs(text: str, top_k: int) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    return [value for value in values if 0 < value <= top_k] or [min(5, top_k)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cutoffs", default="1,3,5,10")
    parser.add_argument("--label-bonus", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_jsonl(args.pred_file)
    rankings_by_split = build_rankings(rows, args.top_k, args.label_bonus)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "pred_file": str(args.pred_file),
        "top_k": args.top_k,
        "label_bonus": args.label_bonus,
        "cutoffs": parse_cutoffs(args.cutoffs, args.top_k),
        "splits": {},
    }
    all_rankings = []
    for split in ["train", "dev", "test", "unknown"]:
        rankings = rankings_by_split.get(split, [])
        if not rankings:
            continue
        write_jsonl(args.output_dir / f"{split}.jsonl", rankings)
        all_rankings.extend(rankings)
        metrics["splits"][split] = ranking_metrics(rankings, metrics["cutoffs"])
    write_jsonl(args.output_dir / "all.jsonl", all_rankings)
    (args.output_dir / "ranking_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
