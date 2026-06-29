"""Tests for OnlineRunner."""

import numpy as np

from rovingbandit import (
    UCB1,
    BanditEnvironment,
    BestArmIdentification,
    EpsilonNeymanAllocation,
    ExploreFirst,
    LinUCB,
    OnlineRunner,
    RegretMinimization,
    ThompsonSampling,
)


def test_basic_run(bernoulli_env):
    runner = OnlineRunner()
    result = runner.run(UCB1(n_arms=5, seed=42), bernoulli_env, n_steps=100)
    assert result.n_steps == 100
    assert len(result.history.arms) == 100
    assert result.total_reward >= 0


def test_objective_metadata_collected(bernoulli_env):
    runner = OnlineRunner()
    obj = RegretMinimization(optimal_reward=0.5)
    result = runner.run(UCB1(n_arms=5, seed=42), bernoulli_env, n_steps=50, objective=obj)
    assert "optimal_reward" in result.metadata
    assert result.final_regret is not None


def test_optimal_reward_added_without_objective(bernoulli_env):
    runner = OnlineRunner()
    result = runner.run(UCB1(n_arms=5, seed=42), bernoulli_env, n_steps=30)
    # no objective -> runner injects optimal_reward from the environment
    assert "optimal_reward" in result.metadata


def test_early_stopping_with_objective_stops_early():
    env = BanditEnvironment(n_arms=3, arm_means=np.array([0.05, 0.1, 0.95]), seed=42)
    runner = OnlineRunner()
    obj = BestArmIdentification(confidence_threshold=0.9, n_mc_samples=300, seed=42)
    result = runner.run(
        ThompsonSampling(n_arms=3, seed=42),
        env,
        n_steps=2000,
        objective=obj,
        early_stopping=True,
    )
    assert result.n_steps < 2000
    assert result.metadata["confidence"] >= 0.9


def test_early_stopping_without_objective_is_noop(bernoulli_env):
    runner = OnlineRunner()
    # early_stopping=True but objective=None: runs all steps without error
    result = runner.run(UCB1(n_arms=5, seed=0), bernoulli_env, n_steps=20, early_stopping=True)
    assert result.n_steps == 20


def test_set_horizon_is_invoked(bernoulli_env):
    runner = OnlineRunner()
    policy = ExploreFirst(n_arms=5, exploration_fraction=0.2, seed=0)
    runner.run(policy, bernoulli_env, n_steps=80)
    assert policy.horizon == 80


def test_runner_sets_horizon_for_neyman(bernoulli_env):
    # EpsilonNeyman raises if horizon unset; the runner must set it
    runner = OnlineRunner()
    policy = EpsilonNeymanAllocation(n_arms=5, exploration_fraction=0.2, seed=0)
    result = runner.run(policy, bernoulli_env, n_steps=60)
    assert result.n_steps == 60


def test_contextual_run(contextual_env):
    runner = OnlineRunner()
    result = runner.run(LinUCB(n_arms=2, alpha=0.1, seed=0), contextual_env, n_steps=40)
    assert result.n_steps == 40
    # contexts were recorded into history
    assert result.history.contexts[-1] is not None


class TestBudget:
    def test_respects_budget(self):
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.1, 0.5, 0.9]),
            costs=np.array([1.0, 2.0, 1.5]),
            seed=42,
        )
        runner = OnlineRunner()
        result = runner.run_with_budget(ThompsonSampling(n_arms=3, seed=42), env, budget=50.0)
        assert result.total_cost <= 50.0
        assert result.metadata["budget"] == 50.0
        assert result.metadata["pay_on_success"] is False

    def test_pay_on_success_only_charges_wins(self):
        # All rewards are 1 -> every pull is charged under pay_on_success
        env = BanditEnvironment(
            n_arms=2,
            arm_means=np.array([1.0, 1.0]),
            costs=np.array([1.0, 1.0]),
            seed=1,
        )
        runner = OnlineRunner()
        result = runner.run_with_budget(
            ThompsonSampling(n_arms=2, seed=1), env, budget=10.0, pay_on_success=True
        )
        assert result.metadata["pay_on_success"] is True
        assert result.total_cost <= 10.0

    def test_runaway_protection_terminates(self):
        # reward always 0 + pay_on_success -> actual_cost 0, budget never decremented.
        # The safety guard (len(history) > budget * 100) must stop the loop.
        env = BanditEnvironment(
            n_arms=2,
            arm_means=np.array([0.0, 0.0]),
            costs=np.array([1.0, 1.0]),
            seed=0,
        )
        runner = OnlineRunner()
        result = runner.run_with_budget(
            ThompsonSampling(n_arms=2, seed=0), env, budget=1.0, pay_on_success=True
        )
        assert 100 <= result.n_steps <= 101
