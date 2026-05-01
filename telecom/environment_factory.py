"""Environment selection factory for telecom models.

This module centralizes environment switching so the RL pipeline can remain
unchanged while selecting between HetNet and Cell-Free telecom models.
"""

from __future__ import annotations

from typing import Any

from constants import ENVIRONMENT_TYPE


def resolve_environment_type(opt: Any = None) -> str:
    """Resolve effective environment type from opt or global constants."""
    env_type = None
    if opt is not None:
        env_type = getattr(opt, "environment_type", None)

    if env_type is None:
        env_type = ENVIRONMENT_TYPE

    env_type = str(env_type).strip().lower()
    if env_type not in {"hetnet", "cell_free"}:
        raise ValueError(
            f"Unsupported environment type: {env_type}. "
            "Supported values are: 'hetnet', 'cell_free'."
        )
    return env_type


def create_scenario(sce: Any, opt: Any = None):
    """Create scenario implementation based on the selected environment."""
    env_type = resolve_environment_type(opt)

    if env_type == "cell_free":
        from telecom_cellfree.cellfree_scenario import CellFreeScenario

        return CellFreeScenario(sce)

    from telecom.scenario import Scenario

    return Scenario(sce)


def create_reward_calculator(
    sce: Any,
    opt: Any,
    device,
    fallback_cls,
):
    """Create reward calculator for the selected environment.

    Args:
        sce: Scenario configuration.
        opt: Optimization/training configuration.
        device: Torch device.
        fallback_cls: HetNet reward calculator class for backward compatibility.
    """
    env_type = resolve_environment_type(opt)

    if env_type == "cell_free":
        from telecom_cellfree.cellfree_reward import CellFreeRewardCalculator

        return CellFreeRewardCalculator(sce, opt, device)

    return fallback_cls(sce, opt, device)
