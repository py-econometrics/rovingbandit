"""Tests for BatchedRunner."""

import numpy as np
import pytest

from rovingbandit import (
    UCB1,
    BanditEnvironment,
    BatchedRunner,
    EpsilonGreedy,
    ExploreFirst,
    RegretMinimization,
)


def test_total_steps_is_batch_times_n_batches(bernoulli_env):
    runner = BatchedRunner()
    result = runner.run(EpsilonGreedy(n_arms=5, epsilon=0.1, seed=42), bernoulli_env, 10, 5)
    assert result.n_steps == 50
    assert result.metadata["batch_size"] == 10
    assert result.metadata["n_batches"] == 5


@pytest.mark.parametrize("batch_size", [1, 7])
def test_various_batch_sizes(bernoulli_env, batch_size):
    runner = BatchedRunner()
    result = runner.run(UCB1(n_arms=5, seed=0), bernoulli_env, batch_size, 4)
    assert result.n_steps == batch_size * 4


def test_set_horizon_uses_total(bernoulli_env):
    runner = BatchedRunner()
    policy = ExploreFirst(n_arms=5, exploration_fraction=0.2, seed=0)
    runner.run(policy, bernoulli_env, batch_size=5, n_batches=6)
    assert policy.horizon == 30


def test_objective_metadata_collected():
    env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), seed=42)
    runner = BatchedRunner()
    obj = RegretMinimization(optimal_reward=0.9)
    result = runner.run(UCB1(n_arms=3, seed=42), env, 5, 10, objective=obj)
    assert "optimal_reward" in result.metadata
    assert result.final_regret is not None
