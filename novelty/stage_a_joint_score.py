from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from .stage_a_joint_base import (
        JOINT_LABELS,
        JointTextConfig,
        JointTextDataset,
        JointTextModel,
        build_loader,
        build_text_lookup,
        class_weights,
        device_from_arg,
        load_novelty_index,
        load_ranking_splits,
        move_to_device,
        ordinal_pos_weights,
        ordinal_targets,
        probabilities,
        read_jsonl,
        sample_train_rows,
        set_seed,
        write_jsonl,
    )
except ImportError:
    from stage_a_joint_base import (
        JOINT_LABELS,
        JointTextConfig,
        JointTextDataset,
        JointTextModel,
        build_loader,
        build_text_lookup,
        class_weights,
        device_from_arg,
        load_novelty_index,
        load_ranking_splits,
        move_to_device,
        ordinal_pos_weights,
        ordinal_targets,
        probabilities,
        read_jsonl,
        sample_train_rows,
        set_seed,
        write_jsonl,
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_thresholds(text: str) -> tuple[float, float, float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--label-thresholds must contain exactly 3 comma-separated values")
    weak, partial, mostly = values
    if not (0.0 <= weak < partial < mostly <= 1.0):
        raise ValueError("--label-thresholds must satisfy 0 <= weak < partial < mostly <= 1")
    return weak, partial, mostly


def score_to_label_ids(score: np.ndarray, thresholds: tuple[float, float, float]) -> np.ndarray:
    weak, partial, mostly = thresholds
    pred = np.zeros(len(score), dtype=np.int64)
    pred[score >= weak] = 1
    pred[score >= partial] = 2
    pred[score >= mostly] = 3
    return pred


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


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0
        out[order[start:end]] = rank
        start = end
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    aa = a.astype(np.float64) - float(a.mean())
    bb = b.astype(np.float64) - float(b.mean())
    denom = float(np.sqrt((aa**2).sum()) * np.sqrt((bb**2).sum()))
    return float((aa * bb).sum() / denom) if denom else 0.0


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    over = np.maximum(0.0, err)
    under = np.maximum(0.0, -err)
    return {
        "mae": round(float(np.abs(err).mean()), 4) if len(err) else 0.0,
        "rmse": round(float(np.sqrt((err**2).mean())), 4) if len(err) else 0.0,
        "pearson": round(corr(y_true, y_pred), 4),
        "spearman": round(corr(ranks(y_true), ranks(y_pred)), 4),
        "mean_over_error": round(float(over.mean()), 4) if len(over) else 0.0,
        "mean_under_error": round(float(under.mean()), 4) if len(under) else 0.0,
    }


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


def label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_class = per_class_metrics(y_true, y_pred)
    f1s = [item["f1"] for item in per_class.values()]
    over_error = np.maximum(0, y_pred - y_true)
    under_error = np.maximum(0, y_true - y_pred)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "ordinal_mae": round(float(np.abs(y_true - y_pred).mean()), 4) if len(y_true) else 0.0,
        "qwk": round(qwk(y_true, y_pred, len(JOINT_LABELS)), 4) if len(y_true) else 0.0,
        "mean_over_error": round(float(over_error.mean()), 4) if len(y_true) else 0.0,
        "mean_under_error": round(float(under_error.mean()), 4) if len(y_true) else 0.0,
        "not_covered_recall": per_class["not_covered"]["recall"],
        "mostly_precision": per_class["mostly_covered"]["precision"],
        "mostly_recall": per_class["mostly_covered"]["recall"],
        "per_class": per_class,
        "pred_distribution": dict(Counter(JOINT_LABELS[int(idx)] for idx in y_pred)),
        "true_distribution": dict(Counter(JOINT_LABELS[int(idx)] for idx in y_true)),
    }


def amp_context(device: torch.device, enabled: bool, bf16: bool) -> torch.autocast:
    device_type = "cuda" if device.type == "cuda" else "mps" if device.type == "mps" else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled and device.type != "cpu")


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    ce_weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["label"].to(outputs["logits"].device)
    scores = batch["score"].to(outputs["logits"].device)
    sample_weight = batch["weight"].to(outputs["logits"].device)
    pred_score = outputs["score"]

    if args.score_loss == "mse":
        score_loss = F.mse_loss(pred_score, scores, reduction="none")
    else:
        score_loss = F.smooth_l1_loss(pred_score, scores, reduction="none", beta=args.huber_beta)
    score_loss = (score_loss * sample_weight).mean()

    over = (F.relu(pred_score - scores) ** 2 * sample_weight).mean()
    under = (F.relu(scores - pred_score) ** 2 * sample_weight).mean()

    ce = F.cross_entropy(outputs["logits"], labels, weight=ce_weights, reduction="none")
    ce = (ce * sample_weight).mean()

    ordinal = F.binary_cross_entropy_with_logits(
        outputs["ordinal_logits"],
        ordinal_targets(labels),
        pos_weight=ordinal_weights,
        reduction="none",
    ).mean(dim=-1)
    ordinal = (ordinal * sample_weight).mean()

    total = (
        args.score_loss_weight * score_loss
        + args.over_coverage_penalty * over
        + args.under_coverage_penalty * under
        + args.ce_loss_weight * ce
        + args.ordinal_loss_weight * ordinal
    )
    return total, {
        "total": float(total.detach().cpu()),
        "score": float(score_loss.detach().cpu()),
        "over_penalty": float(over.detach().cpu()),
        "under_penalty": float(under.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "ordinal": float(ordinal.detach().cpu()),
    }


def metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "neg_score_mae":
        return -float(metrics["score_metrics"]["mae"])
    if name == "score_pearson":
        return float(metrics["score_metrics"]["pearson"])
    if name == "score_spearman":
        return float(metrics["score_metrics"]["spearman"])
    if name == "balanced_score":
        return (
            float(metrics["score_metrics"]["pearson"])
            - 0.5 * float(metrics["score_metrics"]["mae"])
            - 0.25 * float(metrics["score_metrics"]["mean_over_error"])
        )
    return float(metrics["score_decoded_label_metrics"][name])


@torch.no_grad()
def evaluate(
    model: JointTextModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ce_weights: torch.Tensor,
    ordinal_weights: torch.Tensor,
    thresholds: tuple[float, float, float],
    args: argparse.Namespace,
    return_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    y_true_label: list[int] = []
    y_score_true: list[float] = []
    y_score_pred: list[float] = []
    class_argmax: list[int] = []
    prob_all: list[list[float]] = []
    losses: dict[str, list[float]] = defaultdict(list)
    pred_rows: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = move_to_device(batch["inputs"], device)
        with amp_context(device, args.amp, args.bf16):
            outputs = model(inputs)
            loss, parts = compute_loss(outputs, batch, ce_weights, ordinal_weights, args)
        probs = probabilities(outputs, args.probability_mode, args.ordinal_blend_weight)
        class_pred = probs.argmax(dim=-1).detach().cpu().numpy()
        pred_scores = outputs["score"].detach().float().cpu().numpy()
        labels = batch["label"].numpy()
        scores = batch["score"].numpy()
        y_true_label.extend(int(x) for x in labels)
        y_score_true.extend(float(x) for x in scores)
        y_score_pred.extend(float(x) for x in pred_scores)
        class_argmax.extend(int(x) for x in class_pred)
        prob_batch = probs.detach().float().cpu().numpy().tolist()
        prob_all.extend(prob_batch)
        for name, value in parts.items():
            losses[name].append(value)
        losses["loss"].append(float(loss.detach().cpu()))

        if return_predictions:
            score_label_ids = score_to_label_ids(pred_scores, thresholds)
            for idx, target_id in enumerate(batch["target_idea_id"]):
                pred_rows.append(
                    {
                        "target_idea_id": target_id,
                        "gold_joint_coverage_label": JOINT_LABELS[int(labels[idx])],
                        "gold_joint_coverage_score": round(float(scores[idx]), 6),
                        "pred_joint_coverage_label": JOINT_LABELS[int(score_label_ids[idx])],
                        "pred_joint_coverage_score": round(float(pred_scores[idx]), 6),
                        "class_head_pred_joint_coverage_label": JOINT_LABELS[int(class_pred[idx])],
                        "pred_joint_probabilities": {
                            label: round(float(prob_batch[idx][label_idx]), 6)
                            for label_idx, label in enumerate(JOINT_LABELS)
                        },
                        "ranked_priors": batch["ranking"][idx].get("ranked_priors")
                        or batch["ranking"][idx].get("candidate_priors"),
                        "prompt_priors": batch["prompt_priors"][idx],
                    }
                )

    true_label_np = np.asarray(y_true_label, dtype=np.int64)
    score_true_np = np.asarray(y_score_true, dtype=np.float32)
    score_pred_np = np.asarray(y_score_pred, dtype=np.float32)
    score_label_np = score_to_label_ids(score_pred_np, thresholds)
    class_argmax_np = np.asarray(class_argmax, dtype=np.int64)

    metrics = {
        "score_metrics": score_metrics(score_true_np, score_pred_np),
        "score_decoded_label_metrics": label_metrics(true_label_np, score_label_np),
        "class_head_label_metrics": label_metrics(true_label_np, class_argmax_np),
        "loss": {name: round(float(np.mean(values)), 6) for name, values in losses.items()},
    }
    if return_predictions:
        metrics["_predictions"] = pred_rows
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-dir", type=Path, default=None)
    parser.add_argument("--ranking-file", type=Path, default=None)
    parser.add_argument("--train-ranking-file", type=Path, default=None)
    parser.add_argument("--dev-ranking-file", type=Path, default=None)
    parser.add_argument("--test-ranking-file", type=Path, default=None)
    parser.add_argument("--novelty-file", type=Path, default=Path("dataset/idea_novelty_dataset.jsonl"))
    parser.add_argument("--pair-text-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/stage_joint_coverage_score_text_deberta"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--score-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-beta", type=float, default=0.08)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--over-coverage-penalty", type=float, default=0.6)
    parser.add_argument("--under-coverage-penalty", type=float, default=0.15)
    parser.add_argument("--ce-loss-weight", type=float, default=0.05)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.2)
    parser.add_argument("--probability-mode", choices=["class", "ordinal", "blend"], default="blend")
    parser.add_argument("--ordinal-blend-weight", type=float, default=0.25)
    parser.add_argument("--label-thresholds", default="0.25,0.50,0.78")
    parser.add_argument(
        "--selection-metric",
        choices=["neg_score_mae", "score_pearson", "score_spearman", "balanced_score", "qwk", "accuracy", "macro_f1"],
        default="balanced_score",
    )
    parser.add_argument("--include-prior-signals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-summary-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.label_thresholds)
    set_seed(args.seed)

    novelty_index = load_novelty_index(args.novelty_file)
    text_lookup = build_text_lookup(novelty_index, args.pair_text_file)
    split_rows, split_source = load_ranking_splits(args, novelty_index)
    if args.max_train_samples:
        split_rows["train"] = sample_train_rows(split_rows["train"], novelty_index, args.max_train_samples, args.seed)
    if args.max_eval_samples:
        split_rows["dev"] = split_rows["dev"][: args.max_eval_samples]
        split_rows["test"] = split_rows["test"][: args.max_eval_samples]

    datasets = {
        split: JointTextDataset(
            rows,
            novelty_index,
            text_lookup,
            args.top_k,
            args.include_prior_signals,
            args.include_summary_features,
        )
        for split, rows in split_rows.items()
    }

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

    loaders = {
        "train": build_loader(datasets["train"], tokenizer, args.batch_size, args.max_length, True, args.num_workers),
        "dev": build_loader(datasets["dev"], tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers),
        "test": build_loader(datasets["test"], tokenizer, args.eval_batch_size, args.max_length, False, args.num_workers),
    }

    ce_weights = class_weights(datasets["train"], device)
    ordinal_weights = ordinal_pos_weights(datasets["train"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loaders["train"]) * args.epochs)
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
        "model_config": {
            "model_name": cfg.model_name,
            "dropout": cfg.dropout,
            "trust_remote_code": cfg.trust_remote_code,
            "cache_dir": cfg.cache_dir,
            "local_files_only": cfg.local_files_only,
        },
        "labels": JOINT_LABELS,
        "label_thresholds": {
            "weak": thresholds[0],
            "partial": thresholds[1],
            "mostly": thresholds[2],
        },
        "split_source": split_source,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "train_class_distribution": dict(Counter(JOINT_LABELS[int(row["label"])] for row in datasets["train"].items)),
        "class_weights": [round(float(x), 4) for x in ce_weights.detach().cpu().tolist()],
        "ordinal_pos_weights": [round(float(x), 4) for x in ordinal_weights.detach().cpu().tolist()],
        "missing_prior_texts": {split: dataset.missing_prior_texts for split, dataset in datasets.items()},
        "loss_note": "score regression is primary; hard labels are auxiliary and exported labels are decoded from score thresholds",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, list[float]] = defaultdict(list)
        progress = tqdm(loaders["train"], desc=f"train epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            inputs = move_to_device(batch["inputs"], device)
            with amp_context(device, args.amp, args.bf16):
                outputs = model(inputs)
                loss, parts = compute_loss(outputs, batch, ce_weights, ordinal_weights, args)
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
                    score=round(float(np.mean(running["score"][-args.log_every:])), 4),
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        dev_metrics = evaluate(model, loaders["dev"], device, ce_weights, ordinal_weights, thresholds, args)
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
                    "config": run_config,
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
    for split in ["train", "dev", "test"]:
        metrics = evaluate(model, loaders[split], device, ce_weights, ordinal_weights, thresholds, args, return_predictions=True)
        pred_rows = metrics.pop("_predictions")
        final_metrics[split] = metrics
        write_jsonl(args.output_dir / f"{split}_predictions.jsonl", pred_rows)
        print(json.dumps({"split": split, **metrics}, ensure_ascii=False, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
