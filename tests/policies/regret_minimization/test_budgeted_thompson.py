"""Tests for BudgetedThompsonSampling."""

import numpy as np
import pytest

from rovingbandit import BudgetedThompsonSampling, InvalidConfigurationError


def test_costs_default_to_ones():
    policy = BudgetedThompsonSampling(n_arms=3, seed=0)
    assert np.allclose(policy.costs, np.ones(3))


def test_costs_length_mismatch_raises():
    with pytest.raises(InvalidConfigurationError, match="length of costs"):
        BudgetedThompsonSampling(n_arms=3, costs=np.array([1.0, 2.0]))


def test_prefers_better_reward_per_cost_ratio():
    # Arm 0: high reward, high cost (ratio ~0.09); arm 1: med reward, low cost (ratio ~0.5)
    costs = np.array([10.0, 1.0])
    policy = BudgetedThompsonSampling(n_arms=2, costs=costs, seed=42)
    policy.successes = np.array([90.0, 50.0])
    policy.failures = np.array([10.0, 50.0])
    counts = np.zeros(2)
    for _ in range(50):
        counts[policy.select_arm()] += 1
    assert counts[1] > counts[0]


def test_state_roundtrip_includes_costs():
    costs = np.array([2.0, 3.0])
    policy = BudgetedThompsonSampling(n_arms=2, costs=costs, seed=0)
    state = policy.get_state()
    restored = BudgetedThompsonSampling(n_arms=2, seed=1)
    restored.set_state(state)
    assert np.array_equal(restored.costs, costs)
