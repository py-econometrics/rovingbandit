"""Tests for BudgetedUCB."""

import numpy as np

from rovingbandit import BudgetedUCB


def test_initial_costs_zeroed():
    policy = BudgetedUCB(n_arms=3, seed=42)
    assert np.allclose(policy.avg_costs, 0.0)


def test_avg_costs_updated_incrementally():
    policy = BudgetedUCB(n_arms=3, seed=42)
    costs = [1.0, 10.0, 1.0]
    rewards = [1.0, 1.0, 0.0]
    for i in range(3):
        arm = policy.select_arm()
        assert arm == i  # initial phase pulls each arm once in order
        policy.update(arm, rewards[i], costs[i])
    assert np.allclose(policy.avg_costs, [1.0, 10.0, 1.0])
    assert np.allclose(policy.values, [1.0, 1.0, 0.0])


def test_prefers_low_cost_high_value_arm():
    policy = BudgetedUCB(n_arms=3, seed=42)
    costs = [1.0, 10.0, 1.0]
    rewards = [1.0, 1.0, 0.0]
    for i in range(3):
        arm = policy.select_arm()
        policy.update(arm, rewards[i], costs[i])
    # arm 0: high value, low cost -> best budgeted score
    assert policy.select_arm() == 0


def test_state_roundtrip_includes_avg_costs():
    policy = BudgetedUCB(n_arms=2, seed=0)
    policy.update(0, 1.0, 2.0)
    policy.update(1, 0.0, 3.0)
    state = policy.get_state()
    restored = BudgetedUCB(n_arms=2, seed=1)
    restored.set_state(state)
    assert np.array_equal(restored.avg_costs, policy.avg_costs)


def test_reset_clears_avg_costs():
    policy = BudgetedUCB(n_arms=2, seed=0)
    policy.update(0, 1.0, 5.0)
    policy.reset()
    assert np.allclose(policy.avg_costs, 0.0)
