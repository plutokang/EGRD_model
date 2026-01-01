"""Run EGRD: Stage A evidence preparation, then jointly trained Stages B and C."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STEP_ORDER = ("pair", "pair_calibration", "ranking", "joint", "residual")
STEP_DESCRIPTIONS = {
    "pair": "Train Pair Coverage",
    "pair_calibration": "Calibrate Pair Coverage predictions",
    "ranking": "Rank historical priors from Pair Coverage",
    "joint": "Train Joint Coverage MIL v9",
    "residual": "Jointly train Stage B residual decomposition and Stage C novelty prediction",
}

# Paper stages describe the model; execution steps describe training jobs.
PAPER_STAGES = {
    "A": {
        "description": "Historical Evidence Inputs",
        "steps": ["pair", "pair_calibration", "ranking", "joint"],
        "modules": ["stage_a_pair_model.py", "stage_a_calibrate.py", "stage_a_rank.py", "stage_a_joint_mil.py"],
    },
    "B": {
        "description": "Evidence-Grounded Residual Decomposition",
        "steps": ["residual"],
        "modules": ["stage_b_decomposition.py"],
    },
    "C": {
        "description": "Novelty Prediction",
        "steps": ["residual"],
        "modules": ["stage_c_prediction.py"],
    },
}


def boundary_step(name: str, *, stop: bool = False) -> str:
    if name in PAPER_STAGES:
        steps = PAPER_STAGES[name]["steps"]
        return steps[-1] if stop else steps[0]
    return name


ORIGINAL_V23_PROFILE = "original_v23"
EXPLAINABLE_V24_PROFILE = "explainable_v24"
ORIGINAL_V23_DATA_FILES = {
    "idea_novelty_dataset.jsonl": {
        "rows": 5225,
        "sha256": "f6af95d8d852e6468f24b081416db0eda814591b3174d70b9bc7a029145c2c2e",
    },
    "idea_pair_coverage_llm.jsonl": {
        "rows": 21183,
        "sha256": "4b26913a2d3dc2864ca320b58436905fd32eaac3883d7dae7e052239b2307f2f",
    },
    "prior_ranking_dataset.jsonl": {
        "rows": 5225,
        "sha256": "9191caa518cf7c981bec62f2f0720499151e10a602f3dbcf791b9922a28a1127",
    },
}


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    log_path: Path

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.command, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.abspath(base / path))


def find_pipeline_dir(invocation_dir: Path) -> Path:
    required_scripts = (
        Path("stage_a_pair_model.py"),
        Path("stage_a_calibrate.py"),
        Path("stage_a_rank.py"),
        Path("stage_a_joint_base.py"),
        Path("stage_a_joint_score.py"),
        Path("stage_a_joint_mil.py"),
        Path("egrd.py"),
        Path("features.py"),
        Path("stage_b_decomposition.py"),
        Path("stage_c_prediction.py"),
    )
    script_path = Path(__file__)
    logical_script = (
        script_path
        if script_path.is_absolute()
        else Path(os.path.abspath(invocation_dir / script_path))
    )
    raw_candidates = [
        logical_script.parent,
        invocation_dir / "novelty",
        Path(__file__).resolve().parent,
    ]
    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    for candidate in candidates:
        if all((candidate / script).is_file() for script in required_scripts):
            return candidate
    checked_lines = []
    for candidate in candidates:
        missing = [str(script) for script in required_scripts if not (candidate / script).is_file()]
        checked_lines.append(
            f"  - {candidate}\n"
            + "\n".join(f"      missing: {name}" for name in missing)
        )
    raise FileNotFoundError(
        "The novelty directory is incomplete.\n"
        f"Checked directories:\n{chr(10).join(checked_lines)}\n"
        "Copy the complete novelty directory next to the entry point."
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def dataset_file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows += 1
    return {"path": str(path), "rows": rows, "sha256": digest.hexdigest()}


def validate_original_v23_data(data_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    errors: list[str] = []
    for filename, expected in ORIGINAL_V23_DATA_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        actual = dataset_file_manifest(path)
        manifest[filename] = actual
        if actual["rows"] != expected["rows"]:
            errors.append(
                f"{path}: rows={actual['rows']} expected={expected['rows']}"
            )
        if actual["sha256"] != expected["sha256"]:
            errors.append(
                f"{path}: sha256={actual['sha256']} expected={expected['sha256']}"
            )
    if errors:
        raise ValueError(
            "The original_v23 profile requires the retained original dataset:\n  - "
            + "\n  - ".join(errors)
        )
    return manifest


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_text(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def parse_extra_args(text: str, stage: str, protected: set[str]) -> list[str]:
    values = shlex.split(text) if text.strip() else []
    for token in values:
        flag = token.split("=", 1)[0]
        if flag in protected:
            raise ValueError(
                f"{stage} extra arguments may not override pipeline-owned option {flag}. "
                "Use the corresponding pipeline option instead."
            )
    return values


def add_optional_flag(command: list[str], enabled: bool, positive: str, negative: str | None = None) -> None:
    if enabled:
        command.append(positive)
    elif negative is not None:
        command.append(negative)


def all_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def require_paths(paths: Iterable[Path], context: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing inputs for {context}:\n{formatted}")


def run_and_tee(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n[{utc_now()}] {command_text(command)}\n"
        log.write(header)
        log.flush()
        print(header.rstrip(), flush=True)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def choose_pair_split_files(args: argparse.Namespace, invocation_dir: Path) -> tuple[Path, Path, Path] | None:
    explicit = [
        resolve_path(args.pair_train_file, invocation_dir),
        resolve_path(args.pair_dev_file, invocation_dir),
        resolve_path(args.pair_test_file, invocation_dir),
    ]
    if any(explicit):
        if not all(explicit):
            raise ValueError(
                "--pair-train-file, --pair-dev-file, and --pair-test-file "
                "must be provided together"
            )
        return explicit[0], explicit[1], explicit[2]

    if args.pair_split_source == "temporal":
        return None

    pair_staged = args.data_dir / "pair_staged"
    candidates = (
        pair_staged / "idea_pair_coverage_train_natural.jsonl",
        pair_staged / "idea_pair_coverage_dev.jsonl",
        pair_staged / "idea_pair_coverage_test.jsonl",
    )
    require_paths(candidates, "Pair Coverage staged split")
    return candidates


def build_stages(args: argparse.Namespace, pipeline_dir: Path, invocation_dir: Path) -> list[Stage]:
    python_bin = args.python_bin
    model_dir = pipeline_dir
    pair_file = resolve_path(args.pair_file, invocation_dir) or args.data_dir / "idea_pair_coverage_llm.jsonl"
    novelty_file = resolve_path(args.novelty_file, invocation_dir) or args.data_dir / "idea_novelty_dataset.jsonl"
    pair_text_file = (
        resolve_path(args.pair_text_file, invocation_dir)
        or args.data_dir / "idea_pair_coverage_llm.jsonl"
    )
    residual_ranking_file = (
        resolve_path(args.residual_ranking_file, invocation_dir)
        or args.data_dir / "prior_ranking_dataset.jsonl"
    )
    split_files = choose_pair_split_files(args, invocation_dir)

    pair_dir = args.pair_output_dir
    calibration_dir = args.pair_calibration_dir
    ranking_dir = args.ranking_output_dir
    joint_dir = args.joint_output_dir
    residual_dir = args.residual_output_dir
    logs_dir = args.output_root / "logs"

    pair_command = [
        python_bin,
        str(model_dir / "stage_a_pair_model.py"),
    ]
    pair_inputs: list[Path]
    if split_files is not None:
        pair_command.extend(
            [
                "--train-pair-file",
                str(split_files[0]),
                "--dev-pair-file",
                str(split_files[1]),
                "--test-pair-file",
                str(split_files[2]),
            ]
        )
        pair_inputs = list(split_files)
    else:
        pair_command.extend(["--pair-file", str(pair_file), "--use-split-field"])
        pair_inputs = [pair_file]
    pair_command.extend(
        [
            "--output-dir",
            str(pair_dir),
            "--model-name",
            args.model_name,
            "--max-length",
            str(args.pair_max_length),
            "--epochs",
            str(args.pair_epochs),
            "--batch-size",
            str(args.pair_batch_size),
            "--eval-batch-size",
            str(args.pair_eval_batch_size),
            "--lr",
            str(args.pair_lr),
            "--weight-decay",
            "0.01",
            "--warmup-ratio",
            "0.08",
            "--dropout",
            "0.15",
            "--label-schema",
            "staged",
            "--hierarchical-only",
            "--relatedness-loss-weight",
            "0.4",
            "--covering-loss-weight",
            "1.0",
            "--strength-loss-weight",
            "0.8",
            "--pair-expected-loss-weight",
            "0.08",
            "--pos-weight-scale",
            "0.5",
            "--pos-weight-cap",
            "5.0",
            "--sampler",
            args.pair_sampler,
            "--sampler-alpha",
            "1.0",
            "--target-sampler-beta",
            "0.5",
            "--export-score-source",
            "auto",
            "--pair-expected-score-mode",
            "coverage_evidence",
            "--selection-metric",
            "pair_ranking_balanced",
            "--seed",
            str(args.pair_seed),
        ]
    )
    if args.device:
        pair_command.extend(["--device", args.device])
    if args.cache_dir:
        pair_command.extend(["--cache-dir", args.cache_dir])
    add_optional_flag(pair_command, args.local_files_only, "--local-files-only")
    add_optional_flag(pair_command, args.amp, "--amp", "--no-amp")
    add_optional_flag(pair_command, args.bf16, "--bf16", "--no-bf16")
    add_optional_flag(pair_command, args.gradient_checkpointing, "--gradient-checkpointing")
    pair_command.extend(
        parse_extra_args(
            args.pair_extra_args,
            "Pair Coverage",
            {
                "--pair-file",
                "--train-pair-file",
                "--dev-pair-file",
                "--test-pair-file",
                "--output-dir",
                "--model-name",
            },
        )
    )

    pair_predictions = pair_dir / "predictions"
    calibration_command = [
        python_bin,
        str(model_dir / "stage_a_calibrate.py"),
        "--pred-dir",
        str(pair_predictions),
        "--output-dir",
        str(calibration_dir),
        "--objective",
        args.pair_calibration_objective,
        "--threshold-steps",
        str(args.pair_calibration_threshold_steps),
    ]
    calibration_command.extend(
        parse_extra_args(
            args.pair_calibration_extra_args,
            "Pair calibration",
            {"--pred-dir", "--train-file", "--dev-file", "--test-file", "--output-dir"},
        )
    )

    ranking_prediction_file = (
        pair_predictions / "all.jsonl"
        if args.ranking_source == "raw"
        else calibration_dir / "all.jsonl"
    )
    ranking_command = [
        python_bin,
        str(model_dir / "stage_a_rank.py"),
        "--pred-file",
        str(ranking_prediction_file),
        "--output-dir",
        str(ranking_dir),
        "--top-k",
        str(args.ranking_top_k),
        "--cutoffs",
        args.ranking_cutoffs,
        "--label-bonus",
        str(args.ranking_label_bonus),
    ]

    joint_command = [
        python_bin,
        str(model_dir / "stage_a_joint_mil.py"),
        "--ranking-dir",
        str(ranking_dir),
        "--novelty-file",
        str(novelty_file),
        "--pair-text-file",
        str(pair_text_file),
        "--output-dir",
        str(joint_dir),
        "--model-name",
        args.model_name,
        "--top-k",
        str(args.top_k),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.joint_epochs),
        "--batch-size",
        str(args.joint_batch_size),
        "--eval-batch-size",
        str(args.joint_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.joint_gradient_accumulation_steps),
        "--lr",
        str(args.joint_lr),
        "--weight-decay",
        "0.01",
        "--warmup-ratio",
        "0.08",
        "--dropout",
        "0.15",
        "--noisy-or-blend-weight",
        "0.25",
        "--score-loss",
        "mse",
        "--score-loss-target",
        "hierarchical",
        "--score-loss-weight",
        "0.30",
        "--score-prediction-mode",
        "hierarchical",
        "--score-blend-weight",
        "0.30",
        "--hierarchical-loss-weight",
        "1.0",
        "--any-pos-weight",
        "1.30",
        "--substantial-pos-weight",
        "1.0",
        "--mostly-pos-weight",
        "0.70",
        "--lambda-any",
        "1.0",
        "--lambda-substantial",
        "1.0",
        "--lambda-mostly",
        "0.75",
        "--consistency-loss-weight",
        "0.10",
        "--consistency-huber-beta",
        "0.10",
        "--ce-loss-weight",
        "0.15",
        "--ce-class-weight-mode",
        "mild_partial",
        "--partial-ce-weight",
        "1.20",
        "--covered-internal-loss-weight",
        "0.08",
        "--covered-internal-class-weight-mode",
        "sqrt",
        "--covered-internal-max-class-weight",
        "3.0",
        "--ordinal-loss-weight",
        "0.0",
        "--pair-aux-loss-weight",
        "0.15",
        "--pair-aux-target",
        "gold",
        "--include-pred-pair-signals",
        "--prediction-mode",
        "conditional",
        "--ordinal-blend-weight",
        "0.25",
        "--hier-class-blend-weight",
        "0.30",
        "--internal-blend-weight",
        "0.50",
        "--label-thresholds",
        "0.25,0.50,0.78",
        "--covered-internal-any-threshold",
        "0.50",
        "--conditional-thresholds",
        "0.50,0.58,0.53",
        "--tune-conditional-thresholds-on-dev",
        "--threshold-search-metric",
        "balanced",
        "--threshold-steps",
        "81",
        "--conditional-any-range",
        "0.30,0.70",
        "--conditional-substantial-range",
        "0.30,0.70",
        "--conditional-mostly-range",
        "0.50,0.62",
        "--threshold-min-not-recall",
        "0.70",
        "--threshold-min-weak-recall",
        "0.10",
        "--threshold-min-partial-recall",
        "0.20",
        "--threshold-min-mostly-recall",
        "0.35",
        "--threshold-min-covered-recall",
        "0.70",
        "--threshold-max-partial-pred-ratio",
        "0.22",
        "--threshold-max-mostly-pred-ratio",
        "0.30",
        "--threshold-max-partial-pred-multiplier",
        "3.0",
        "--threshold-max-mostly-pred-multiplier",
        "2.5",
        "--balanced-sampling",
        "sqrt",
        "--selection-metric",
        "selected_constrained_balanced_score",
        "--seed",
        str(args.joint_seed),
    ]
    if args.device:
        joint_command.extend(["--device", args.device])
    if args.cache_dir:
        joint_command.extend(["--cache-dir", args.cache_dir])
    add_optional_flag(joint_command, args.local_files_only, "--local-files-only")
    add_optional_flag(joint_command, args.amp, "--amp", "--no-amp")
    add_optional_flag(joint_command, args.bf16, "--bf16", "--no-bf16")
    add_optional_flag(joint_command, args.gradient_checkpointing, "--gradient-checkpointing")
    joint_command.extend(
        parse_extra_args(
            args.joint_extra_args,
            "Joint Coverage",
            {
                "--ranking-dir",
                "--ranking-file",
                "--novelty-file",
                "--pair-text-file",
                "--output-dir",
                "--model-name",
            },
        )
    )

    residual_command = [
        python_bin,
        str(model_dir / "egrd.py"),
        "--novelty-file",
        str(novelty_file),
        "--joint-pred-dir",
        str(joint_dir),
        "--pair-text-file",
        str(pair_text_file),
        "--output-dir",
        str(residual_dir),
        "--model-name",
        args.model_name,
        "--top-k",
        str(args.top_k),
        "--num-semantic-units",
        str(args.num_semantic_units),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.residual_epochs),
        "--batch-size",
        str(args.residual_batch_size),
        "--eval-batch-size",
        str(args.residual_eval_batch_size),
        "--lr",
        str(args.residual_lr),
        "--encoder-lr-ratio",
        "0.2",
        "--freeze-encoder-epochs",
        "1",
        "--unfreeze-last-layers",
        "4",
        "--clip-grad-norm",
        "1.0",
        "--amp-dtype",
        args.residual_amp_dtype,
        "--coverage-feature-noise-std",
        "0.03",
        "--coverage-feature-dropout",
        "0.10",
        "--combination-pos-weight",
        "3.0",
        "--moderate-high-pos-weight",
        "1.8",
        "--hard-negative-weight",
        "1.5",
    ]
    if args.pipeline_profile == ORIGINAL_V23_PROFILE:
        residual_command.extend(
            [
                "--semantic-assignment-mode",
                "legacy",
                "--semantic-diversity-loss-weight",
                "0.05",
                "--attention-overlap-loss-weight",
                "0.70",
                "--query-diversity-loss-weight",
                "0.10",
                "--unit-usage-balance-loss-weight",
                "0.20",
                "--slot-entropy-loss-weight",
                "0.0",
                "--semantic-unit-separation-loss-weight",
                "0.0",
                "--unit-collapse-loss-weight",
                "0.02",
                "--min-unit-coverage-variance",
                "0.0025",
                "--semantic-selection-weight",
                "0.0",
                "--semantic-top-token-count",
                "6",
                "--selection-metric",
                "novelty_selection_score_constrained",
                "--final-checkpoint",
                "novelty",
            ]
        )
        residual_checkpoint_dir = "best_novelty"
    else:
        residual_command.extend(
            [
                "--semantic-assignment-mode",
                "sparsemax",
                "--semantic-score-temperature",
                "0.07",
                "--semantic-score-smoothing-kernel",
                "3",
                "--semantic-sparsemax-weight",
                "0.95",
                "--semantic-balance-iterations",
                "3",
                "--semantic-min-slot-mass",
                "0.25",
                "--semantic-diversity-loss-weight",
                "0.10",
                "--attention-overlap-loss-weight",
                "0.35",
                "--slot-entropy-loss-weight",
                "0.30",
                "--semantic-unit-separation-loss-weight",
                "0.20",
                "--max-semantic-unit-cosine",
                "0.80",
                "--unit-usage-balance-loss-weight",
                "0.10",
                "--query-diversity-loss-weight",
                "0.05",
                "--unit-collapse-loss-weight",
                "0.05",
                "--min-unit-coverage-variance",
                "0.005",
                "--semantic-selection-weight",
                "0.12",
                "--semantic-top-token-count",
                "6",
                "--selection-metric",
                "novelty_explainable_selection_score_constrained",
                "--final-checkpoint",
                "novelty_explainable",
            ]
        )
        residual_checkpoint_dir = "best_novelty_explainable"
    residual_command.extend(["--seed", str(args.residual_seed)])
    if args.residual_ranking_source == "dataset":
        residual_command.extend(["--ranking-file", str(residual_ranking_file)])
        residual_ranking_inputs: tuple[Path, ...] = (residual_ranking_file,)
    else:
        residual_command.extend(["--ranking-dir", str(ranking_dir)])
        residual_ranking_inputs = tuple(
            ranking_dir / f"{split}.jsonl" for split in ("train", "dev", "test")
        )
    residual_init_checkpoint = resolve_path(args.residual_init_checkpoint, invocation_dir)
    if residual_init_checkpoint is not None:
        residual_command.extend(
            [
                "--init-from-checkpoint",
                str(residual_init_checkpoint),
                "--reinit-semantic-modules",
                "--freeze-regression-heads-epochs",
                "1",
            ]
        )
        residual_init_inputs = [residual_init_checkpoint]
    else:
        joint_encoder_checkpoint = joint_dir / "best" / "checkpoint.pt"
        residual_command.extend(["--encoder-checkpoint", str(joint_encoder_checkpoint)])
        residual_init_inputs = [joint_encoder_checkpoint]
    if args.device:
        residual_command.extend(["--device", args.device])
    add_optional_flag(residual_command, args.amp, "--amp")
    add_optional_flag(residual_command, args.gradient_checkpointing, "--gradient-checkpointing")
    residual_command.extend(
        parse_extra_args(
            args.residual_extra_args,
            "Residual EGRD",
            {
                "--ranking-dir",
                "--ranking-file",
                "--novelty-file",
                "--joint-pred-dir",
                "--pair-text-file",
                "--output-dir",
                "--model-name",
                "--encoder-checkpoint",
                "--init-from-checkpoint",
            },
        )
    )

    ranking_split_outputs = tuple(ranking_dir / f"{split}.jsonl" for split in ("train", "dev", "test"))
    joint_prediction_outputs = tuple(
        joint_dir / f"{split}_predictions.jsonl" for split in ("train", "dev", "test")
    )
    stages = [
        Stage(
            "pair",
            pair_command,
            tuple(pair_inputs),
            (
                pair_dir / "best" / "checkpoint.pt",
                pair_predictions / "train.jsonl",
                pair_predictions / "dev.jsonl",
                pair_predictions / "test.jsonl",
                pair_predictions / "all.jsonl",
                pair_dir / "metrics.json",
            ),
            logs_dir / "01_pair.log",
        ),
        Stage(
            "pair_calibration",
            calibration_command,
            (
                pair_predictions / "train.jsonl",
                pair_predictions / "dev.jsonl",
                pair_predictions / "test.jsonl",
            ),
            (
                calibration_dir / "pair_calibration.json",
                calibration_dir / "train.jsonl",
                calibration_dir / "dev.jsonl",
                calibration_dir / "test.jsonl",
                calibration_dir / "all.jsonl",
            ),
            logs_dir / "02_pair_calibration.log",
        ),
        Stage(
            "ranking",
            ranking_command,
            (ranking_prediction_file,),
            ranking_split_outputs
            + (
                ranking_dir / "all.jsonl",
                ranking_dir / "ranking_metrics.json",
            ),
            logs_dir / "03_ranking.log",
        ),
        Stage(
            "joint",
            joint_command,
            ranking_split_outputs + (novelty_file, pair_text_file),
            (
                joint_dir / "best" / "checkpoint.pt",
                joint_dir / "metrics.json",
            )
            + joint_prediction_outputs,
            logs_dir / "04_joint.log",
        ),
        Stage(
            "residual",
            residual_command,
            residual_ranking_inputs
            + (
                novelty_file,
                pair_text_file,
            )
            + joint_prediction_outputs
            + tuple(residual_init_inputs),
            (
                residual_dir / "metrics.json",
                residual_dir / "all_predictions.jsonl",
                residual_dir / residual_checkpoint_dir / "checkpoint.pt",
            ),
            logs_dir / "05_residual.log",
        ),
    ]
    return stages


def artifact_manifest(args: argparse.Namespace) -> dict[str, Any]:
    residual_checkpoint_dir = (
        "best_novelty"
        if args.pipeline_profile == ORIGINAL_V23_PROFILE
        else "best_novelty_explainable"
    )
    return {
        "preset": args.pipeline_profile,
        "pair_checkpoint": str(args.pair_output_dir / "best" / "checkpoint.pt"),
        "pair_predictions": str(args.pair_output_dir / "predictions"),
        "calibrated_pair_predictions": str(args.pair_calibration_dir),
        "prior_rankings": str(args.ranking_output_dir),
        "joint_checkpoint": str(args.joint_output_dir / "best" / "checkpoint.pt"),
        "joint_predictions": str(args.joint_output_dir),
        "residual_checkpoint": str(
            args.residual_output_dir / residual_checkpoint_dir / "checkpoint.pt"
        ),
        "residual_predictions": str(args.residual_output_dir / "all_predictions.jsonl"),
        "residual_ranking_source": args.residual_ranking_source,
        "metrics": {
            "pair": str(args.pair_output_dir / "metrics.json"),
            "pair_calibration": str(args.pair_calibration_dir / "pair_calibration.json"),
            "ranking": str(args.ranking_output_dir / "ranking_metrics.json"),
            "joint": str(args.joint_output_dir / "metrics.json"),
            "residual": str(args.residual_output_dir / "metrics.json"),
        },
    }


def make_config(args: argparse.Namespace, stages: list[Stage]) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "pipeline_version": 4,
        "preset": args.pipeline_profile,
        "staged_training": True,
        "note": (
            "Stage A prepares fixed evidence features and a joint-coverage checkpoint. "
            "Stages B and C are optimized jointly; gradients propagate from C through B, "
            "but not into the Stage A models."
        ),
        "data_dir": str(args.data_dir),
        "dataset_manifest": args.dataset_manifest,
        "output_root": str(args.output_root),
        "model_name": args.model_name,
        "top_k": args.top_k,
        "ranking_source": args.ranking_source,
        "pair_split_source": args.pair_split_source,
        "residual_ranking_source": args.residual_ranking_source,
        "stages": PAPER_STAGES,
        "steps": {
            stage.name: {
                "description": STEP_DESCRIPTIONS[stage.name],
                "command": stage.command,
                "command_text": command_text(stage.command),
                "fingerprint": stage.fingerprint,
                "inputs": [str(path) for path in stage.inputs],
                "outputs": [str(path) for path in stage.outputs],
                "log": str(stage.log_path),
            }
            for stage in stages
        },
        "artifacts": artifact_manifest(args),
    }


def stage_range(args: argparse.Namespace) -> tuple[int, int]:
    start = STEP_ORDER.index(boundary_step(args.start_stage))
    stop = STEP_ORDER.index(boundary_step(args.stop_stage, stop=True))
    if start > stop:
        raise ValueError("--start-stage must not come after --stop-stage")
    return start, stop


def run_pipeline(args: argparse.Namespace, stages: list[Stage], execution_dir: Path) -> int:
    start, stop = stage_range(args)
    selected_stages = stages[start : stop + 1]
    state_path = args.output_root / "pipeline_state.json"
    state = read_json(state_path)
    state["pipeline_version"] = 4
    state.setdefault("created_at", utc_now())
    state.setdefault("stages", {})
    state["status"] = "running"
    state["updated_at"] = utc_now()
    state["artifacts"] = artifact_manifest(args)
    write_json(state_path, state)

    force_index = (
        STEP_ORDER.index(boundary_step(args.force_from_stage))
        if args.force_from_stage is not None
        else len(STEP_ORDER)
    )

    for stage in selected_stages:
        index = STEP_ORDER.index(stage.name)
        previous = state["stages"].get(stage.name, {})
        matching_run = previous.get("fingerprint") == stage.fingerprint
        outputs_exist = all_exist(stage.outputs)
        forced = index >= force_index

        if args.skip_completed and not forced and matching_run and outputs_exist:
            print(f"[skip step] {stage.name}: completed outputs already exist", flush=True)
            continue
        if (
            args.skip_completed
            and not forced
            and previous
            and previous.get("status") == "completed"
            and not matching_run
        ):
            raise RuntimeError(
                f"Stage {stage.name} was completed with a different command. "
                f"Use --force-from-stage {stage.name} to rerun it and downstream stages."
            )
        if args.skip_completed and not forced and args.adopt_existing and outputs_exist:
            print(f"[adopt] {stage.name}: using complete existing artifacts", flush=True)
            state["stages"][stage.name] = {
                "status": "completed",
                "adopted": True,
                "fingerprint": stage.fingerprint,
                "command": stage.command,
                "outputs": [str(path) for path in stage.outputs],
                "completed_at": utc_now(),
            }
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            continue

        require_paths(stage.inputs, stage.name)
        stage_record = {
            "status": "running",
            "description": STEP_DESCRIPTIONS[stage.name],
            "fingerprint": stage.fingerprint,
            "command": stage.command,
            "command_text": command_text(stage.command),
            "inputs": [str(path) for path in stage.inputs],
            "outputs": [str(path) for path in stage.outputs],
            "log": str(stage.log_path),
            "started_at": utc_now(),
        }
        state["stages"][stage.name] = stage_record
        state["updated_at"] = utc_now()
        write_json(state_path, state)

        print(f"\n=== {STEP_DESCRIPTIONS[stage.name]} ===", flush=True)
        return_code = run_and_tee(stage.command, execution_dir, stage.log_path)
        stage_record["return_code"] = return_code
        if return_code != 0:
            stage_record["status"] = "failed"
            stage_record["failed_at"] = utc_now()
            state["status"] = "failed"
            state["failed_stage"] = stage.name
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            raise RuntimeError(
                f"Stage {stage.name} failed with exit code {return_code}. "
                f"See {stage.log_path}"
            )
        require_paths(stage.outputs, f"{stage.name} outputs")
        stage_record["status"] = "completed"
        stage_record["completed_at"] = utc_now()
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    full_run_complete = stop == len(STEP_ORDER) - 1 and all(
        state["stages"].get(name, {}).get("status") == "completed" for name in STEP_ORDER
    )
    state["status"] = "completed" if full_run_complete else "partial_completed"
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    write_json(
        args.output_root / "pipeline_manifest.json",
        {
            "status": state["status"],
            "updated_at": state["updated_at"],
            "artifacts": artifact_manifest(args),
            "stage_status": {
                name: state["stages"].get(name, {}).get("status", "not_run")
                for name in STEP_ORDER
            },
        },
    )
    print(f"\nPipeline status: {state['status']}", flush=True)
    print(f"Manifest: {args.output_root / 'pipeline_manifest.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline-profile",
        choices=[ORIGINAL_V23_PROFILE, EXPLAINABLE_V24_PROFILE],
        default=ORIGINAL_V23_PROFILE,
        help="Formal original v2.3 profile or the later explainable v2.4 experiment.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset"),
        help=(
            "Directory containing the three original dataset files. The "
            "original_v23 profile validates their row counts and SHA-256 hashes."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/runs/original_v23_deberta_top5"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-semantic-units", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use BF16 in Pair and Joint. The validated formal run used "
            "FP16 there, so this defaults to false. Residual precision is controlled "
            "separately by --residual-amp-dtype."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    data = parser.add_argument_group("data overrides")
    data.add_argument("--pair-file", type=Path, default=None)
    data.add_argument("--pair-text-file", type=Path, default=None)
    data.add_argument("--novelty-file", type=Path, default=None)
    data.add_argument("--pair-train-file", type=Path, default=None)
    data.add_argument("--pair-dev-file", type=Path, default=None)
    data.add_argument("--pair-test-file", type=Path, default=None)
    data.add_argument(
        "--pair-split-source",
        choices=["temporal", "staged"],
        default="temporal",
        help=(
            "temporal reproduces the validated run using idea_pair_coverage_llm.jsonl; "
            "staged uses pair_staged/idea_pair_coverage_{train_natural,dev,test}.jsonl."
        ),
    )

    outputs = parser.add_argument_group("stage output overrides")
    outputs.add_argument("--pair-output-dir", type=Path, default=None)
    outputs.add_argument("--pair-calibration-dir", type=Path, default=None)
    outputs.add_argument("--ranking-output-dir", type=Path, default=None)
    outputs.add_argument("--joint-output-dir", type=Path, default=None)
    outputs.add_argument("--residual-output-dir", type=Path, default=None)

    pair = parser.add_argument_group("Stage A: pair coverage")
    pair.add_argument("--pair-epochs", type=int, default=4)
    pair.add_argument("--pair-batch-size", type=int, default=4)
    pair.add_argument("--pair-eval-batch-size", type=int, default=16)
    pair.add_argument("--pair-max-length", type=int, default=768)
    pair.add_argument("--pair-lr", type=float, default=1e-5)
    pair.add_argument("--pair-seed", type=int, default=83)
    pair.add_argument(
        "--pair-sampler",
        choices=["none", "label", "sqrt_label", "covering"],
        default="sqrt_label",
    )
    pair.add_argument(
        "--pair-extra-args",
        default="",
        help="Quoted additional hyperparameters passed to stage_a_pair_model.py",
    )

    calibration = parser.add_argument_group("Stage A: pair calibration and ranking")
    calibration.add_argument(
        "--pair-calibration-objective",
        choices=["balanced", "macro_f1", "qwk", "accuracy", "covering_f1"],
        default="balanced",
    )
    calibration.add_argument("--pair-calibration-threshold-steps", type=int, default=41)
    calibration.add_argument("--pair-calibration-extra-args", default="")
    calibration.add_argument(
        "--ranking-source",
        choices=["raw", "calibrated"],
        default="raw",
        help="Raw pair_expected matches the current successful Joint v9 setup.",
    )
    calibration.add_argument("--ranking-top-k", type=int, default=5)
    calibration.add_argument("--ranking-cutoffs", default="1,3,5")
    calibration.add_argument("--ranking-label-bonus", type=float, default=0.02)

    joint = parser.add_argument_group("Stage A: joint coverage")
    joint.add_argument("--joint-epochs", type=int, default=5)
    joint.add_argument("--joint-batch-size", type=int, default=2)
    joint.add_argument("--joint-eval-batch-size", type=int, default=8)
    joint.add_argument("--joint-gradient-accumulation-steps", type=int, default=8)
    joint.add_argument("--joint-lr", type=float, default=1e-5)
    joint.add_argument("--joint-seed", type=int, default=257)
    joint.add_argument("--joint-extra-args", default="")

    residual = parser.add_argument_group("Stages B + C: residual decomposition and novelty prediction")
    residual.add_argument("--residual-epochs", type=int, default=10)
    residual.add_argument("--residual-batch-size", type=int, default=2)
    residual.add_argument("--residual-eval-batch-size", type=int, default=8)
    residual.add_argument("--residual-lr", type=float, default=1.5e-5)
    residual.add_argument("--residual-seed", type=int, default=173)
    residual.add_argument(
        "--residual-amp-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
    )
    residual.add_argument(
        "--residual-ranking-source",
        choices=["dataset", "pipeline"],
        default="dataset",
        help=(
            "dataset reproduces the validated run with prior_ranking_dataset.jsonl; "
            "pipeline feeds the newly generated Pair ranking into Residual."
        ),
    )
    residual.add_argument(
        "--residual-ranking-file",
        type=Path,
        default=None,
        help="Override prior_ranking_dataset.jsonl when --residual-ranking-source=dataset.",
    )
    residual.add_argument(
        "--residual-init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional EGRD warm-start. By default the residual encoder is initialized "
            "from the Joint Coverage best checkpoint."
        ),
    )
    residual.add_argument("--residual-extra-args", default="")

    execution = parser.add_argument_group("pipeline execution")
    execution.add_argument("--start-stage", choices=(*PAPER_STAGES, *STEP_ORDER), default="A",
                           help="Paper stage or execution step. B and C share one joint training job.")
    execution.add_argument("--stop-stage", choices=(*PAPER_STAGES, *STEP_ORDER), default="C",
                           help="B/C both include the joint decomposition and prediction job.")
    execution.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip outputs completed by this pipeline with the same command.",
    )
    execution.add_argument(
        "--adopt-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust and skip complete artifacts that were not recorded by this pipeline.",
    )
    execution.add_argument(
        "--force-from-stage",
        choices=(*PAPER_STAGES, *STEP_ORDER),
        default=None,
        help="Rerun this stage and every selected downstream stage.",
    )
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source inputs and print all commands without starting training.",
    )
    return parser


def normalize_args(args: argparse.Namespace, invocation_dir: Path) -> None:
    args.data_dir = resolve_path(args.data_dir, invocation_dir)
    args.output_root = resolve_path(args.output_root, invocation_dir)
    assert args.data_dir is not None
    assert args.output_root is not None
    if args.pipeline_profile == ORIGINAL_V23_PROFILE:
        args.dataset_manifest = validate_original_v23_data(args.data_dir)
    else:
        args.dataset_manifest = {
            filename: dataset_file_manifest(args.data_dir / filename)
            for filename in ORIGINAL_V23_DATA_FILES
            if (args.data_dir / filename).is_file()
        }
    args.pair_output_dir = resolve_path(args.pair_output_dir, invocation_dir) or args.output_root / "pair_coverage"
    args.pair_calibration_dir = (
        resolve_path(args.pair_calibration_dir, invocation_dir)
        or args.output_root / "pair_calibration"
    )
    args.ranking_output_dir = (
        resolve_path(args.ranking_output_dir, invocation_dir)
        or args.output_root / "prior_rankings"
    )
    args.joint_output_dir = (
        resolve_path(args.joint_output_dir, invocation_dir)
        or args.output_root / "joint_coverage"
    )
    args.residual_output_dir = (
        resolve_path(args.residual_output_dir, invocation_dir)
        or args.output_root / "residual_egrd"
    )
    if args.ranking_top_k < args.top_k:
        raise ValueError("--ranking-top-k must be greater than or equal to --top-k")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logical_pwd = os.environ.get("PWD")
    invocation_dir = (
        Path(logical_pwd).expanduser()
        if logical_pwd and Path(logical_pwd).is_absolute()
        else Path.cwd()
    )
    try:
        pipeline_dir = find_pipeline_dir(invocation_dir)
        normalize_args(args, invocation_dir)
        stages = build_stages(args, pipeline_dir, invocation_dir)
        start, stop = stage_range(args)
        selected_stages = stages[start : stop + 1]
        config = make_config(args, stages)
        if args.dry_run:
            source_inputs: list[Path] = []
            produced: set[Path] = set()
            for stage in selected_stages:
                source_inputs.extend(path for path in stage.inputs if path not in produced)
                produced.update(stage.outputs)
            require_paths(source_inputs, "pipeline source data")
            print(json.dumps(config, ensure_ascii=False, indent=2))
            return 0

        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "pipeline_config.json", config)
        return run_pipeline(args, stages, invocation_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
