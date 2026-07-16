"""Tests for History and Result (including plotting)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rovingbandit import History, InvalidConfigurationError, Result


def make_result(arms, rewards, costs=None, metadata=None, n_arms=None):
    """Build a Result from explicit observations."""
    history = History()
    costs = costs if costs is not None else [0.0] * len(arms)
    for arm, reward, cost in zip(arms, rewards, costs, strict=True):
        history.add(arm, reward, cost)
    n_arms = n_arms if n_arms is not None else (max(arms) + 1 if arms else 1)
    values = np.zeros(n_arms)
    for arm, reward in zip(arms, rewards, strict=True):
        values[arm] = reward
    policy_state = {"n_arms": n_arms, "values": values}
    return Result(history=history, policy_state=policy_state, metadata=metadata or {})


class TestHistory:
    def test_add_and_len(self):
        h = History()
        assert len(h) == 0
        h.add(0, 1.0, 0.5)
        h.add(1, 0.0)
        assert len(h) == 2
        assert h.arms == [0, 1]
        assert h.rewards == [1.0, 0.0]
        assert h.costs == [0.5, 0.0]

    def test_arrays_and_cumulative(self):
        h = History()
        for r in [1.0, 0.0, 1.0, 1.0]:
            h.add(0, r, 2.0)
        assert np.array_equal(h.rewards_array, np.array([1.0, 0.0, 1.0, 1.0]))
        assert np.array_equal(h.cumulative_rewards, np.array([1.0, 1.0, 2.0, 3.0]))
        assert np.array_equal(h.cumulative_costs, np.array([2.0, 4.0, 6.0, 8.0]))

    def test_average_reward_over_time(self):
        h = History()
        for r in [1.0, 0.0, 1.0]:
            h.add(0, r)
        # cumulative [1,1,2] / steps [1,2,3]
        assert np.allclose(h.average_reward, np.array([1.0, 0.5, 2.0 / 3.0]))


class TestResultProperties:
    def test_basic_aggregates(self):
        res = make_result(arms=[0, 1, 1], rewards=[1.0, 0.0, 1.0], costs=[1.0, 2.0, 2.0])
        assert res.n_steps == 3
        assert res.total_reward == 2.0
        assert res.total_cost == 5.0
        assert np.isclose(res.average_reward, 2.0 / 3.0)

    def test_average_reward_empty_is_zero(self):
        res = make_result(arms=[], rewards=[], n_arms=2)
        assert res.n_steps == 0
        assert res.average_reward == 0.0

    def test_best_arm(self):
        res = make_result(arms=[0, 1, 2], rewards=[0.2, 0.9, 0.5])
        assert res.best_arm == 1

    def test_cumulative_regret_none_without_optimal(self):
        res = make_result(arms=[0, 1], rewards=[1.0, 0.0])
        assert res.cumulative_regret is None
        assert res.final_regret is None

    def test_cumulative_regret_with_optimal(self):
        res = make_result(
            arms=[0, 0, 0],
            rewards=[0.0, 1.0, 0.0],
            metadata={"optimal_reward": 1.0},
        )
        # optimal - reward = [1, 0, 1] -> cumsum [1, 1, 2]
        assert np.array_equal(res.cumulative_regret, np.array([1.0, 1.0, 2.0]))
        assert res.final_regret == 2.0

    def test_metadata_backed_properties(self):
        res = make_result(
            arms=[0],
            rewards=[1.0],
            metadata={
                "confidence": 0.95,
                "group_shares": np.array([0.4, 0.6]),
                "estimation_variance": 0.123,
            },
        )
        assert res.confidence == 0.95
        assert np.array_equal(res.group_shares, np.array([0.4, 0.6]))
        assert res.estimation_variance == 0.123

    def test_missing_metadata_properties_are_none(self):
        res = make_result(arms=[0], rewards=[1.0])
        assert res.confidence is None
        assert res.group_shares is None
        assert res.estimation_variance is None


class TestSummary:
    def test_summary_minimal(self):
        res = make_result(arms=[0, 1], rewards=[1.0, 0.0])
        summary = res.summary()
        assert summary["n_steps"] == 2
        assert summary["total_reward"] == 1.0
        assert "best_arm" in summary
        assert "final_regret" not in summary
        assert "confidence" not in summary

    def test_summary_includes_optional_keys(self):
        res = make_result(
            arms=[0, 0],
            rewards=[1.0, 0.0],
            metadata={
                "optimal_reward": 1.0,
                "confidence": 0.9,
                "group_shares": np.array([1.0]),
                "estimation_variance": 0.5,
            },
        )
        summary = res.summary()
        assert "final_regret" in summary
        assert summary["confidence"] == 0.9
        assert "group_shares" in summary
        assert summary["estimation_variance"] == 0.5


class TestPlot:
    @pytest.mark.parametrize(
        ("metric", "ylabel"),
        [
            ("cumulative_reward", "Cumulative Reward"),
            ("average_reward", "Average Reward"),
            ("arm_pulls", "Cumulative Pulls"),
        ],
    )
    def test_plot_returns_axes_with_labels(self, metric, ylabel):
        res = make_result(arms=[0, 1, 0, 2], rewards=[1.0, 0.0, 1.0, 0.5], n_arms=3)
        ax = res.plot(metric=metric)
        assert ax is not None
        assert ax.get_ylabel() == ylabel
        assert ax.get_xlabel() == "Steps"

    def test_plot_cumulative_regret(self):
        res = make_result(arms=[0, 0], rewards=[0.0, 1.0], metadata={"optimal_reward": 1.0})
        ax = res.plot(metric="cumulative_regret")
        assert ax.get_ylabel() == "Cumulative Regret"

    def test_plot_reuses_passed_axes(self):
        res = make_result(arms=[0, 1], rewards=[1.0, 0.0])
        fig, ax = plt.subplots()
        returned = res.plot(metric="cumulative_reward", ax=ax)
        assert returned is ax

    def test_plot_regret_without_optimal_raises(self):
        res = make_result(arms=[0, 1], rewards=[1.0, 0.0])
        with pytest.raises(InvalidConfigurationError, match="cumulative regret unavailable"):
            res.plot(metric="cumulative_regret")

    def test_plot_unknown_metric_raises(self):
        res = make_result(arms=[0, 1], rewards=[1.0, 0.0])
        with pytest.raises(InvalidConfigurationError, match="unknown metric"):
            res.plot(metric="not_a_metric")
