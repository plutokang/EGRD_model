from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PAIR_LABELS = ["not_covering", "related_not_covering", "partial_cover", "large_cover"]
PAIR_TO_ID = {label: idx for idx, label in enumerate(PAIR_LABELS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    for idx, label in enumerate(PAIR_LABELS):
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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_class = per_class_metrics(y_true, y_pred)
    f1s = [item["f1"] for item in per_class.values()]
    covering_true = y_true >= 2
    covering_pred = y_pred >= 2
    covering_tp = int((covering_true & covering_pred).sum())
    covering_fp = int((~covering_true & covering_pred).sum())
    covering_fn = int((covering_true & ~covering_pred).sum())
    covering_precision = covering_tp / max(1, covering_tp + covering_fp)
    covering_recall = covering_tp / max(1, covering_tp + covering_fn)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "ordinal_mae": round(float(np.abs(y_true - y_pred).mean()), 4) if len(y_true) else 0.0,
        "qwk": round(qwk(y_true, y_pred, len(PAIR_LABELS)), 4) if len(y_true) else 0.0,
        "covering_precision": round(float(covering_precision), 4),
        "covering_recall": round(float(covering_recall), 4),
        "per_class": per_class,
        "pred_distribution": dict(Counter(PAIR_LABELS[int(idx)] for idx in y_pred)),
        "true_distribution": dict(Counter(PAIR_LABELS[int(idx)] for idx in y_true)),
    }


def metric_score(metrics: dict[str, Any], objective: str) -> float:
    if objective == "macro_f1":
        return float(metrics["macro_f1"])
    if objective == "qwk":
        return float(metrics["qwk"])
    if objective == "accuracy":
        return float(metrics["accuracy"])
    if objective == "covering_f1":
        precision = float(metrics["covering_precision"])
        recall = float(metrics["covering_recall"])
        return 2 * precision * recall / max(1e-12, precision + recall)
    return 0.5 * float(metrics["macro_f1"]) + 0.5 * float(metrics["qwk"])


def fast_search_score(y_true: np.ndarray, y_pred: np.ndarray, objective: str) -> tuple[float, float, float]:
    n_classes = len(PAIR_LABELS)
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    cm = np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    macro_f1 = float(f1.mean())
    accuracy = float(tp.sum() / max(1.0, cm.sum()))

    weights = np.zeros_like(cm)
    denom = max(1, (n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / denom
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / max(1.0, cm.sum())
    den = float((weights * expected).sum())
    qwk_value = float(1 - (weights * cm).sum() / den) if den else 0.0

    covering_true = y_true >= 2
    covering_pred = y_pred >= 2
    covering_tp = float((covering_true & covering_pred).sum())
    covering_fp = float((~covering_true & covering_pred).sum())
    covering_fn = float((covering_true & ~covering_pred).sum())
    covering_precision = covering_tp / max(1.0, covering_tp + covering_fp)
    covering_recall = covering_tp / max(1.0, covering_tp + covering_fn)
    covering_f1 = 2 * covering_precision * covering_recall / max(1e-12, covering_precision + covering_recall)

    if objective == "macro_f1":
        score = macro_f1
    elif objective == "qwk":
        score = qwk_value
    elif objective == "accuracy":
        score = accuracy
    elif objective == "covering_f1":
        score = covering_f1
    else:
        score = 0.5 * macro_f1 + 0.5 * qwk_value
    return float(score), float(covering_precision), float(covering_recall)


def label_ids(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([PAIR_TO_ID.get(str(row.get(key) or "not_covering"), 0) for row in rows], dtype=np.int64)


def related_probs(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row.get("pred_relatedness_probability") or 0.0) for row in rows], dtype=np.float32)


def coverage_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row.get("pred_coverage_score") or 0.0) for row in rows], dtype=np.float32)


def direct_expected_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        probs = row.get("pred_pair_probabilities") or {}
        expected = sum(idx * float(probs.get(label, 0.0)) for idx, label in enumerate(PAIR_LABELS))
        values.append(expected / max(1, len(PAIR_LABELS) - 1))
    return np.asarray(values, dtype=np.float32)


def blend_scores(coverage: np.ndarray, expected: np.ndarray, direct_weight: float) -> np.ndarray:
    weight = min(1.0, max(0.0, float(direct_weight)))
    return (1.0 - weight) * coverage + weight * expected


def raw_pred_ids(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        label = row.get("pred_pair_label")
        if label in PAIR_TO_ID:
            values.append(PAIR_TO_ID[str(label)])
            continue
        probs = row.get("pred_pair_probabilities") or {}
        values.append(int(np.argmax([float(probs.get(label, 0.0)) for label in PAIR_LABELS])))
    return np.asarray(values, dtype=np.int64)


def candidate_thresholds(values: np.ndarray, steps: int, lo: float, hi: float) -> list[float]:
    if len(values) == 0:
        return [0.5]
    quantiles = np.linspace(0.02, 0.98, max(3, steps))
    candidates = set(float(np.quantile(values, q)) for q in quantiles)
    candidates.update(float(x) for x in np.linspace(lo, hi, max(3, steps)))
    clipped = [min(hi, max(lo, value)) for value in candidates]
    ordered = sorted(set(round(value, 6) for value in clipped))
    if len(ordered) <= steps:
        return ordered
    keep = np.linspace(0, len(ordered) - 1, steps).round().astype(int)
    return [ordered[int(idx)] for idx in keep]


def apply_thresholds(
    related: np.ndarray,
    score: np.ndarray,
    related_threshold: float,
    partial_threshold: float,
    large_threshold: float,
) -> np.ndarray:
    pred = np.zeros(len(score), dtype=np.int64)
    is_related = related >= related_threshold
    pred[is_related] = 1
    pred[is_related & (score >= partial_threshold)] = 2
    pred[is_related & (score >= large_threshold)] = 3
    return pred


def tune_thresholds(
    y_true: np.ndarray,
    related: np.ndarray,
    coverage: np.ndarray,
    expected: np.ndarray,
    objective: str,
    threshold_steps: int,
    blend_weights: list[float],
    min_covering_precision: float,
    min_covering_recall: float,
) -> dict[str, Any]:
    related_candidates = candidate_thresholds(related, threshold_steps, 0.05, 0.95)
    best: dict[str, Any] | None = None
    for blend_weight in blend_weights:
        score = blend_scores(coverage, expected, blend_weight)
        score_candidates = candidate_thresholds(score, threshold_steps, 0.02, 0.98)
        for related_threshold in related_candidates:
            for partial_threshold in score_candidates:
                for large_threshold in score_candidates:
                    if large_threshold <= partial_threshold:
                        continue
                    pred = apply_thresholds(related, score, related_threshold, partial_threshold, large_threshold)
                    score_value, covering_precision, covering_recall = fast_search_score(y_true, pred, objective)
                    if covering_precision < min_covering_precision:
                        continue
                    if covering_recall < min_covering_recall:
                        continue
                    if best is None or score_value > best["objective_score"]:
                        metrics = classification_metrics(y_true, pred)
                        best = {
                            "objective": objective,
                            "objective_score": round(float(score_value), 6),
                            "min_covering_precision": round(float(min_covering_precision), 6),
                            "min_covering_recall": round(float(min_covering_recall), 6),
                            "direct_score_weight": round(float(blend_weight), 6),
                            "related_threshold": round(float(related_threshold), 6),
                            "partial_score_threshold": round(float(partial_threshold), 6),
                            "large_score_threshold": round(float(large_threshold), 6),
                            "dev_metrics": metrics,
                        }
    if best is None:
        score = blend_scores(coverage, expected, 0.0)
        pred = apply_thresholds(related, score, 0.5, 0.45, 0.72)
        best = {
            "objective": objective,
            "objective_score": round(float(metric_score(classification_metrics(y_true, pred), objective)), 6),
            "min_covering_precision": round(float(min_covering_precision), 6),
            "min_covering_recall": round(float(min_covering_recall), 6),
            "direct_score_weight": 0.0,
            "related_threshold": 0.5,
            "partial_score_threshold": 0.45,
            "large_score_threshold": 0.72,
            "dev_metrics": classification_metrics(y_true, pred),
            "fallback": True,
            "fallback_reason": "no threshold combination satisfied the covering precision/recall constraints",
        }
    return best


def calibrated_ids(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    related = related_probs(rows)
    coverage = coverage_scores(rows)
    expected = direct_expected_scores(rows)
    score = blend_scores(coverage, expected, float(calibration["direct_score_weight"]))
    pred = apply_thresholds(
        related,
        score,
        float(calibration["related_threshold"]),
        float(calibration["partial_score_threshold"]),
        float(calibration["large_score_threshold"]),
    )
    return pred, score


def apply_to_rows(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    pred, score = calibrated_ids(rows, calibration)
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["raw_pred_pair_label"] = row.get("pred_pair_label")
        item["calibrated_pair_label"] = PAIR_LABELS[int(pred[idx])]
        item["calibrated_pair_label_id"] = int(pred[idx])
        item["calibrated_score"] = round(float(score[idx]), 6)
        item["calibration_version"] = "pair_threshold_calibration_v1"
        out.append(item)
    return out


def default_output_dir(pred_dir: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    if pred_dir is not None:
        return pred_dir.parent / "calibrated_pair_predictions"
    return Path("results/runs/calibrated_pair_predictions")


def parse_blend_weights(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=None, help="Directory containing train/dev/test JSONL predictions.")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--objective", choices=["balanced", "macro_f1", "qwk", "accuracy", "covering_f1"], default="balanced")
    parser.add_argument("--threshold-steps", type=int, default=41)
    parser.add_argument("--blend-weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--min-covering-precision", type=float, default=0.0)
    parser.add_argument("--min-covering-recall", type=float, default=0.0)
    args = parser.parse_args()

    if args.pred_dir is not None:
        train_file = args.pred_dir / "train.jsonl"
        dev_file = args.pred_dir / "dev.jsonl"
        test_file = args.pred_dir / "test.jsonl"
    else:
        train_file, dev_file, test_file = args.train_file, args.dev_file, args.test_file
    if train_file is None or dev_file is None or test_file is None:
        raise ValueError("Provide --pred-dir or all of --train-file/--dev-file/--test-file")

    train_rows = read_jsonl(train_file)
    dev_rows = read_jsonl(dev_file)
    test_rows = read_jsonl(test_file)

    dev_true = label_ids(dev_rows, "gold_pair_label")
    calibration = tune_thresholds(
        dev_true,
        related_probs(dev_rows),
        coverage_scores(dev_rows),
        direct_expected_scores(dev_rows),
        args.objective,
        args.threshold_steps,
        parse_blend_weights(args.blend_weights),
        args.min_covering_precision,
        args.min_covering_recall,
    )

    output_dir = default_output_dir(args.pred_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {"train": train_rows, "dev": dev_rows, "test": test_rows}
    metrics: dict[str, Any] = {
        "calibration": calibration,
        "source_files": {
            "train": str(train_file),
            "dev": str(dev_file),
            "test": str(test_file),
        },
        "raw": {},
        "calibrated": {},
    }
    all_calibrated = []
    for split, rows in split_rows.items():
        y_true = label_ids(rows, "gold_pair_label")
        raw_pred = raw_pred_ids(rows)
        calibrated_pred, _ = calibrated_ids(rows, calibration)
        metrics["raw"][split] = classification_metrics(y_true, raw_pred)
        metrics["calibrated"][split] = classification_metrics(y_true, calibrated_pred)
        calibrated = apply_to_rows(rows, calibration)
        write_jsonl(output_dir / f"{split}.jsonl", calibrated)
        all_calibrated.extend(calibrated)
    write_jsonl(output_dir / "all.jsonl", all_calibrated)
    (output_dir / "pair_calibration.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
