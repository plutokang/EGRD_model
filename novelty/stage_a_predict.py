from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_a_pair_model import (
    BASE_PAIR_LABELS,
    PairCoverageCollator,
    PairCoverageConfig,
    PairCoverageDataset,
    PairCoverageModel,
    pair_score_levels_from_rows,
    predict,
    prediction_rows,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-file", type=Path, required=True, help="Rows with target_text / prior_text / dates")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained pair-coverage checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="run_config.json of the training run. Strongly recommended: model-name, max-length, "
        "label-schema and direct-blend-weight must match the checkpoint, and a mismatch produces "
        "plausible but wrong scores rather than an error.",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--label-schema", choices=["staged", "ordinal"], default="staged")
    parser.add_argument("--direct-blend-weight", type=float, default=0.35)
    parser.add_argument("--score-source", default="pair_expected", help="Which score prediction_rows exports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--split-name",
        default="unknown",
        help="Value written to each row's split field. stage_a_rank only "
        "emits splits named train/dev/test/unknown, so anything else silently produces an empty "
        "all.jsonl. 'unknown' also keeps the EGRD on its own temporal split rather than forcing "
        "every row into one bucket and leaving the others empty.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the first N pairs (0 = all). For checking the checkpoint loads and the "
        "environment matches before committing to the full run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run_config is not None:
        cfg = json.loads(args.run_config.read_text(encoding="utf-8"))
        scopes = [cfg] + [cfg[key] for key in ("args", "model_config") if isinstance(cfg.get(key), dict)]

        def lookup(key: str) -> Any:
            for scope in scopes:
                if scope.get(key) is not None:
                    return scope[key]
            return None

        resolved = []
        for flag, key in (
            ("model_name", "model_name"),
            ("max_length", "max_length"),
            ("label_schema", "label_schema"),
            ("direct_blend_weight", "direct_blend_weight"),
            ("dropout", "dropout"),
            ("score_source", "effective_score_source"),
        ):
            value = lookup(key)
            if value is not None:
                setattr(args, flag, value)
                resolved.append(flag)
        missing = [f for f in ("model_name", "max_length", "label_schema", "direct_blend_weight") if f not in resolved]
        if missing:
            print(f"error: {args.run_config} has no value for {missing}. Running with defaults would "
                  "produce plausible but wrong scores; pass them explicitly instead.", file=sys.stderr)
            raise SystemExit(2)
        print(f"config from {args.run_config}: model={args.model_name} max_length={args.max_length} "
              f"label_schema={args.label_schema} direct_blend={args.direct_blend_weight} "
              f"dropout={args.dropout} score_source={args.score_source}")
    return args


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.pair_file)
    if not rows:
        print(f"error: {args.pair_file} is empty", file=sys.stderr)
        return 1
    indices = list(range(len(rows)))
    if args.limit:
        indices = indices[: args.limit]
        print(f"--limit {args.limit}: scoring {len(indices)} of {len(rows)} pairs")

    out_path = args.output_dir / "pair_predictions.jsonl"
    if out_path.exists() and not args.overwrite:
        print(f"error: {out_path} exists; pass --overwrite", file=sys.stderr)
        return 2

    pair_labels = list(BASE_PAIR_LABELS)
    pair_score_levels = pair_score_levels_from_rows(rows, indices, pair_labels, "coverage_evidence")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = PairCoverageModel(
        PairCoverageConfig(
            model_name=args.model_name,
            dropout=args.dropout,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    ).float().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warning: load_state_dict missing={list(missing)} unexpected={list(unexpected)}", file=sys.stderr)

    loader = DataLoader(
        PairCoverageDataset(rows, indices, pair_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PairCoverageCollator(tokenizer, args.max_length),
    )
    print(f"scoring {len(indices)} pairs from {args.pair_file} on {device}")
    pred = predict(
        model, loader, device, pair_labels, pair_score_levels,
        args.label_schema, args.direct_blend_weight, args.amp, args.bf16,
    )

    out_rows = prediction_rows(rows, args.split_name, pred, pair_labels, args.score_source)
    for out_row, row in zip(out_rows, (rows[i] for i in pred["row_idx"])):
        out_row["target_text"] = row.get("target_text")
        out_row["prior_text"] = row.get("prior_text")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out_rows)

    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["pred_pair_label"]] = counts.get(row["pred_pair_label"], 0) + 1
    scores = sorted(row["pred_coverage_score"] for row in out_rows)
    summary: dict[str, Any] = {
        "pair_file": str(args.pair_file),
        "checkpoint": str(args.checkpoint),
        "rows": len(out_rows),
        "pred_label_distribution": counts,
        "pred_coverage_score": {
            "median": scores[len(scores) // 2],
            "p25": scores[len(scores) // 4],
            "p75": scores[3 * len(scores) // 4],
        },
        "pair_score_levels": pair_score_levels,
        "label_schema": args.label_schema,
        "score_source": args.score_source,
    }
    (args.output_dir / "pair_prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\npredicted label distribution: {counts}")
    print(f"coverage score p25/median/p75: {summary['pred_coverage_score']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
