# MLflow experiments

Track PyMC model runs with configs, posteriors, and LOO comparison.

## Run an experiment

From the project root (with `.venv` activated):

```bash
python -m src.experiments experiments/configs/baseline.yaml
```

Quick smoke test (few draws):

```bash
python -m src.experiments experiments/configs/quick_test.yaml
```

## Browse runs

```bash
mlflow ui
```

Open http://127.0.0.1:5000 and select experiment **world-cup-scores**. Sort by `loo_elpd` (higher is better).

## Compare runs in Python

Open [`notebooks/compare_experiments.ipynb`](../notebooks/compare_experiments.ipynb).

## Add a new variant

1. Copy an existing file in `configs/`.
2. Change `name` and model/data/sample settings.
3. Run: `python -m src.experiments experiments/configs/your_config.yaml`

## Artifacts per run

| File | Contents |
|------|----------|
| `trace.nc` | Posterior (xarray DataTree / ArviZ format) |
| `config.yaml` | Exact config used |
| `summary.csv` | Parameter summary statistics |
| `loo.json` | LOO scores (if computable) |

Local copies also land in `experiments/runs/<run_id>/` (gitignored).

## Environment

Set tracking URI (optional; defaults to `sqlite:///mlflow.db` in the project root):

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

See [`.env.example`](../.env.example).
