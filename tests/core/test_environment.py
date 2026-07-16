"""Tests for BanditEnvironment."""

import numpy as np
import pytest

from rovingbandit import (
    BanditEnvironment,
    InvalidArmError,
    InvalidConfigurationError,
    MissingConfigurationError,
)


class TestConstruction:
    """Construction and input validation."""

    def test_defaults_costs_to_ones(self):
        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]))
        assert env.n_arms == 3
        assert np.allclose(env.costs, np.ones(3))

    def test_custom_costs_stored(self):
        costs = np.array([1.0, 2.0, 1.5])
        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), costs=costs)
        assert np.allclose(env.costs, costs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"n_arms": 3, "arm_means": np.array([0.1, 0.2])},
                "arm_means has length 2",
            ),
            (
                {"n_arms": 2, "arm_means": np.array([0.1, 1.5])},
                r"arm_means must lie in \[0, 1\]",
            ),
            (
                {"n_arms": 2, "arm_means": np.array([0.1, -0.1])},
                r"arm_means must lie in \[0, 1\]",
            ),
            (
                {
                    "n_arms": 3,
                    "arm_means": np.array([0.1, 0.5, 0.9]),
                    "costs": np.array([1.0, 2.0]),
                },
                "costs has length 2",
            ),
            (
                {
                    "n_arms": 2,
                    "arm_means": np.array([0.1, 0.5]),
                    "costs": np.array([1.0, 0.0]),
                },
                "costs must be strictly positive",
            ),
            (
                {
                    "n_arms": 2,
                    "arm_means": np.array([0.1, 0.5]),
                    "arm_groups": np.array([0, 1, 0]),
                },
                "arm_groups has length 3",
            ),
            (
                {
                    "n_arms": 2,
                    "arm_means": np.array([0.1, 0.5]),
                    "contexts": np.ones((3, 4)),
                },
                "contexts first dimension is 3",
            ),
        ],
    )
    def test_invalid_construction_raises(self, kwargs, match):
        with pytest.raises(InvalidConfigurationError, match=match):
            BanditEnvironment(**kwargs)


class TestPull:
    """Reward/cost sampling."""

    def test_bernoulli_reward_and_cost(self):
        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), seed=42)
        reward, cost = env.pull(1)
        assert reward in (0.0, 1.0)
        assert cost == 1.0

    def test_reward_fn_path(self):
        def reward_fn(arm, rng):
            return float(arm * 10)

        env = BanditEnvironment(n_arms=3, arm_means=None, reward_fn=reward_fn, seed=0)
        reward, _ = env.pull(2)
        assert reward == 20.0

    def test_custom_cost_returned(self):
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.1, 0.5, 0.9]),
            costs=np.array([1.0, 2.0, 1.5]),
            seed=1,
        )
        _, cost = env.pull(1)
        assert cost == 2.0

    @pytest.mark.parametrize("arm", [-1, 3, 99])
    def test_out_of_range_arm_raises(self, arm):
        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]))
        with pytest.raises(InvalidArmError, match="out of range"):
            env.pull(arm)

    def test_no_means_or_fn_raises(self):
        env = BanditEnvironment(n_arms=2, arm_means=None)
        with pytest.raises(MissingConfigurationError, match="arm_means or reward_fn"):
            env.pull(0)


class TestOptimal:
    """Optimal arm/reward queries."""

    def test_optimal_arm_reward(self):
        env = BanditEnvironment(n_arms=5, arm_means=np.array([0.1, 0.2, 0.5, 0.4, 0.3]))
        assert env.get_optimal_arm() == 2
        assert env.get_optimal_reward() == 0.5

    def test_optimal_arm_cost(self):
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.1, 0.5, 0.9]),
            costs=np.array([3.0, 1.0, 2.0]),
        )
        assert env.get_optimal_arm(objective="cost") == 1

    def test_optimal_arm_reward_per_cost(self):
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.2, 0.5, 0.9]),
            costs=np.array([1.0, 10.0, 10.0]),
        )
        # reward/cost: 0.2, 0.05, 0.09 -> arm 0
        assert env.get_optimal_arm(objective="reward_per_cost") == 0

    def test_unknown_objective_raises(self):
        env = BanditEnvironment(n_arms=2, arm_means=np.array([0.1, 0.5]))
        with pytest.raises(InvalidConfigurationError, match="unknown objective"):
            env.get_optimal_arm(objective="nonsense")

    def test_optimal_arm_without_means_raises(self):
        env = BanditEnvironment(n_arms=2, arm_means=None, reward_fn=lambda a, r: 0.0)
        with pytest.raises(MissingConfigurationError, match="optimal arm"):
            env.get_optimal_arm()

    def test_optimal_reward_without_means_raises(self):
        env = BanditEnvironment(n_arms=2, arm_means=None, reward_fn=lambda a, r: 0.0)
        with pytest.raises(MissingConfigurationError, match="optimal reward"):
            env.get_optimal_reward()


class TestRngAndContexts:
    """Seeding and context handling."""

    def test_seed_reproducibility(self):
        means = np.array([0.3, 0.5, 0.7])
        env_a = BanditEnvironment(n_arms=3, arm_means=means, seed=7)
        env_b = BanditEnvironment(n_arms=3, arm_means=means, seed=7)
        pulls_a = [env_a.pull(i % 3)[0] for i in range(30)]
        pulls_b = [env_b.pull(i % 3)[0] for i in range(30)]
        assert pulls_a == pulls_b

    def test_reset_rng_restores_sequence(self):
        env = BanditEnvironment(n_arms=2, arm_means=np.array([0.5, 0.5]), seed=3)
        first = [env.pull(0)[0] for _ in range(10)]
        env.reset_rng(3)
        second = [env.pull(0)[0] for _ in range(10)]
        assert first == second

    def test_get_contexts_returns_copy(self):
        contexts = np.array([[1.0, 0.0], [0.0, 1.0]])
        env = BanditEnvironment(n_arms=2, arm_means=np.array([0.1, 0.5]), contexts=contexts)
        returned = env.get_contexts()
        assert np.array_equal(returned, contexts)
        returned[0, 0] = 999.0
        assert env.contexts[0, 0] == 1.0  # original untouched

    def test_get_contexts_none_when_unset(self):
        env = BanditEnvironment(n_arms=2, arm_means=np.array([0.1, 0.5]))
        assert env.get_contexts() is None
