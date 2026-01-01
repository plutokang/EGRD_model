# Evidence-Grounded Residual Decomposition (EGRD)

The code follows the paper's three-stage methodology:

```
Stage A: Historical Evidence Inputs
         pair coverage -> calibration / prior ranking -> joint coverage
                                  |
                       fixed evidence features
                                  v
Stage B: Evidence-Grounded Residual Decomposition
         semantic units -> unit residuals + interaction residuals
                                  |
                                  v
Stage C: Novelty Prediction
         gated context fusion -> auxiliary heads -> novelty branches
         -> development-tuned mixture and moderate-to-high refinement
         -> low / weak / moderate / high
```

Stage A prepares historical evidence, pair features and joint-coverage
predictions. Stage B constructs semantic units, estimates their support with
noisy-OR, and computes unit and interaction residuals. Stage C combines these
representations with evidence context to predict an ordered novelty grade.

**Stages B and C are trained jointly**, as in the paper. Their encoder is
initialized from the Stage A joint-coverage checkpoint; Stage A predictions
remain fixed. The three methodological stages therefore do not correspond to
three independently trained models.

Model and training code only; no data, checkpoints or evaluation outputs.

## Layout

```
novelty/
  run_pipeline.py             orchestrates Stage A and joint B+C training
  stage_a_pair_model.py       pair coverage cross-encoder
  stage_a_predict.py          inference for a trained pair checkpoint
  stage_a_calibrate.py        probability / threshold calibration
  stage_a_rank.py             Top-K prior ranking
  stage_a_joint_base.py       joint coverage base model
  stage_a_joint_score.py      score-first joint coverage variant
  stage_a_joint_mil.py        MIL joint coverage model (default)
  stage_b_decomposition.py    semantic units, support and residuals
  stage_c_prediction.py       gated fusion, auxiliary and novelty heads
  egrd.py                    model assembly, joint B+C training, evaluation,
                             development-set tuning and grade refinement
  features.py                shared labels and feature schema
run.sh
requirements.txt
```

`EGRDResidualPredictor.forward()` calls `decompose()` (Stage B), then
`predict_novelty()` (Stage C), with gradients flowing through both. Model
parameters retain their original names for state-dict checkpoint compatibility.

## Usage

```bash
cd anon_release
pip install -r requirements.txt   # install PyTorch separately for your hardware
bash run.sh
python novelty/run_pipeline.py --help

# Validate inputs and print commands without training.
bash run.sh --dry-run

# Run only historical evidence preparation.
bash run.sh --stop-stage A

# Train B+C using existing Stage A artifacts.
bash run.sh --start-stage B
```

Paths supplied to `run.sh` are relative to the caller's working directory.
`--data-dir` must contain `idea_pair_coverage_llm.jsonl`,
`prior_ranking_dataset.jsonl` and `idea_novelty_dataset.jsonl`; results are
written under `--output-root`. The default `original_v23` profile retains the
original hyperparameters and validates the original dataset's row counts and
SHA-256 hashes. The optional `explainable_v24` profile is a later experimental
variant, not the paper's default semantic-assignment configuration.

`--start-stage`, `--stop-stage` and `--force-from-stage` accept `A`, `B`, or
`C`. Selecting either B or C executes their shared training job. The execution
step names `pair`, `pair_calibration`, `ranking`, `joint`, and `residual` are
also accepted for finer control and resuming previous runs. Configuration
files list the three paper stages under `stages` and training commands under
`steps`; execution state remains indexed by step name.

The default `--residual-ranking-source dataset` retains the validated run's
precomputed rankings. Use `--residual-ranking-source pipeline` to feed the
new Stage A rankings into B+C. Stage A scripts and `egrd.py` also run
standalone; B and C are differentiable model modules.
