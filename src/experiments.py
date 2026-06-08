"""MLflow experiment runner for PyMC football score models."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import arviz as az
import arviz_stats as azstats
import mlflow
import numpy as np
import pandas as pd
import pymc as pm
import yaml

from .data import PROJECT_ROOT, load_modeling_arrays
from .models import build_poisson_model

# PyTensor macOS compiler fix (must run before PyMC sampling)
import pytensor_macos  # noqa: F401, E402


DEFAULT_EXPERIMENT = "world-cup-scores"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested config for MLflow params (string values only)."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=full_key))
        elif value is None:
            continue
        else:
            flat[full_key] = str(value)
    return flat


def _count_divergences(idata) -> int:
    if "sample_stats" not in idata:
        return 0
    diverging = idata["sample_stats"].get("diverging")
    if diverging is None:
        return 0
    return int(np.asarray(diverging).sum())


def _max_rhat(idata) -> float:
    try:
        rhat = azstats.rhat(idata, method="rank")
        values = np.asarray(rhat.to_array().values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.max()) if finite.size else float("nan")
    except Exception:
        return float("nan")


def _min_ess_bulk(idata) -> float:
    try:
        ess = azstats.ess(idata, method="bulk")
        values = np.asarray(ess.to_array().values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.min()) if finite.size else float("nan")
    except Exception:
        return float("nan")


def log_pymc_run(
    idata,
    config: dict[str, Any],
    sampling_time_s: float,
    artifact_dir: Path,
) -> None:
    """Log diagnostics, LOO, and artifacts to the active MLflow run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    trace_path = artifact_dir / "trace.nc"
    idata.to_netcdf(trace_path)
    mlflow.log_artifact(str(trace_path))

    config_path = artifact_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    mlflow.log_artifact(str(config_path))

    summary_df = azstats.summary(idata)
    summary_path = artifact_dir / "summary.csv"
    summary_df.to_csv(summary_path)
    mlflow.log_artifact(str(summary_path))

    mlflow.log_metric("sampling_time_s", sampling_time_s)
    mlflow.log_metric("divergences", _count_divergences(idata))
    mlflow.log_metric("max_rhat", _max_rhat(idata))
    mlflow.log_metric("min_ess_bulk", _min_ess_bulk(idata))

    try:
        loo_home = az.loo(idata, var_name="home_goals", pointwise=False)
        loo_away = az.loo(idata, var_name="away_goals", pointwise=False)
        elpd_total = float(loo_home.elpd + loo_away.elpd)
        se_total = float((loo_home.se**2 + loo_away.se**2) ** 0.5)
        p_total = float(loo_home.p + loo_away.p)
        mlflow.log_metric("loo_elpd", elpd_total)
        mlflow.log_metric("loo_se", se_total)
        mlflow.log_metric("loo_p", p_total)
        mlflow.log_metric("loo_elpd_home", float(loo_home.elpd))
        mlflow.log_metric("loo_elpd_away", float(loo_away.elpd))
        loo_path = artifact_dir / "loo.json"
        loo_path.write_text(
            json.dumps(
                {
                    "elpd_loo": elpd_total,
                    "se": se_total,
                    "p_loo": p_total,
                    "home_goals": {"elpd": float(loo_home.elpd), "se": float(loo_home.se)},
                    "away_goals": {"elpd": float(loo_away.elpd), "se": float(loo_away.se)},
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(loo_path))
    except Exception as exc:
        mlflow.log_param("loo_error", str(exc)[:250])


def run_experiment(
    config_path: Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
    data_path: Path | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Run one configured experiment and log results to MLflow."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI") is None:
        default_db = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{default_db}")

    mlflow.set_experiment(experiment_name)
    run_name = config.get("name", config_path.stem)

    arrays = load_modeling_arrays(config, data_path=data_path)
    sample_kwargs = dict(config.get("sample", {}))
    sample_kwargs.setdefault("return_inferencedata", True)
    sample_kwargs.pop("idata_kwargs", None)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(config))
        mlflow.log_param("config_file", str(config_path))
        mlflow.log_metric("n_matches", arrays.n_matches)
        mlflow.log_metric("n_dropped_missing_scores", arrays.n_dropped_missing_scores)
        mlflow.log_metric("num_teams", arrays.num_teams)

        model = build_poisson_model(config, arrays)
        start = time.perf_counter()
        idata = pm.sample(model=model, **sample_kwargs)
        idata = pm.compute_log_likelihood(idata, model=model)
        sampling_time_s = time.perf_counter() - start

        artifact_dir = PROJECT_ROOT / "experiments" / "runs" / mlflow.active_run().info.run_id
        log_pymc_run(idata, config, sampling_time_s, artifact_dir)

        run_id = mlflow.active_run().info.run_id
        print(f"Finished run '{run_name}' -> run_id={run_id}")
        return run_id


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments <path/to/config.yaml>")
        sys.exit(1)
    run_experiment(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
