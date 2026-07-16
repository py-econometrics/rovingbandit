"""Shared pytest fixtures and configuration for the rovingbandit test suite."""

from collections.abc import Callable

import matplotlib

# Use a non-interactive backend so Result.plot tests never try to open a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rovingbandit import BanditEnvironment  # noqa: E402

# Canonical seed used across tests for deterministic, reproducible behavior.
SEED = 42

# A fixed 5-arm Bernoulli problem whose unique best arm is arm 4 (mean 0.5).
DEFAULT_ARM_MEANS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound figure count."""
    yield
    plt.close("all")


@pytest.fixture
def arm_means() -> np.ndarray:
    """Default Bernoulli arm means (best arm is index 4)."""
    return DEFAULT_ARM_MEANS.copy()


@pytest.fixture
def bernoulli_env(arm_means: np.ndarray) -> BanditEnvironment:
    """A seeded 5-arm Bernoulli environment."""
    return BanditEnvironment(n_arms=5, arm_means=arm_means, seed=SEED)


@pytest.fixture
def env_factory() -> Callable[..., BanditEnvironment]:
    """Return a builder for seeded environments with overridable parameters."""

    def _make(
        n_arms: int = 5,
        arm_means: np.ndarray | None = DEFAULT_ARM_MEANS,
        seed: int | None = SEED,
        **kwargs,
    ) -> BanditEnvironment:
        means = None if arm_means is None else np.asarray(arm_means, dtype=float)
        return BanditEnvironment(n_arms=n_arms, arm_means=means, seed=seed, **kwargs)

    return _make


@pytest.fixture
def contextual_env() -> BanditEnvironment:
    """A 2-arm linear-reward environment for contextual (LinUCB) tests.

    Arm 0 has the higher expected reward under the fixed contexts.
    """
    contexts = np.array([[1.0, 0.0], [0.0, 1.0]])
    theta = np.array([1.0, 0.5])

    def reward_fn(arm: int, rng: np.random.Generator) -> float:
        return float(np.dot(contexts[arm], theta))

    return BanditEnvironment(
        n_arms=2,
        arm_means=None,
        contexts=contexts,
        reward_fn=reward_fn,
        seed=SEED,
    )
