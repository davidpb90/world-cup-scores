"""PyMC model definitions for international football score prediction."""

from __future__ import annotations

from typing import Any

import pymc as pm

from .data import ModelingArrays


def build_poisson_model(
    config: dict[str, Any],
    arrays: ModelingArrays,
) -> pm.Model:
    """Build a Bayesian Poisson model with team attack/defense ratings."""
    model_config = config.get("model", {})
    num_teams = arrays.num_teams

    attack_mu = float(model_config.get("attack_mu", 0.0))
    attack_sigma = float(model_config.get("attack_sigma", 1.5))
    defense_mu = float(model_config.get("defense_mu", 0.0))
    defense_sigma = float(model_config.get("defense_sigma", 1.5))
    home_adv_mu = float(model_config.get("home_advantage_mu", 0.2))
    home_adv_sigma = float(model_config.get("home_advantage_sigma", 0.5))
    use_home_advantage = bool(model_config.get("home_advantage", True))

    with pm.Model() as model:
        attack = pm.Normal(
            "attack", mu=attack_mu, sigma=attack_sigma, shape=num_teams
        )
        defense = pm.Normal(
            "defense", mu=defense_mu, sigma=defense_sigma, shape=num_teams
        )

        if use_home_advantage:
            home_advantage = pm.Normal(
                "home_advantage", mu=home_adv_mu, sigma=home_adv_sigma
            )
            log_theta_home = (
                home_advantage
                + attack[arrays.home_team_array]
                - defense[arrays.away_team_array]
            )
        else:
            log_theta_home = (
                attack[arrays.home_team_array]
                - defense[arrays.away_team_array]
            )

        theta_home = pm.math.exp(log_theta_home)
        theta_away = pm.math.exp(
            attack[arrays.away_team_array] - defense[arrays.home_team_array]
        )

        pm.Poisson("home_goals", mu=theta_home, observed=arrays.home_goals_array)
        pm.Poisson("away_goals", mu=theta_away, observed=arrays.away_goals_array)

    return model
