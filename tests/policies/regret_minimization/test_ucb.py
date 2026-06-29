"""Tests for UCB1."""

import numpy as np

from rovingbandit import UCB1


def test_initial_phase_pulls_each_arm_once():
    policy = UCB1(n_arms=4, seed=42)
    pulled = []
    for _ in range(4):
        arm = policy.select_arm()
        pulled.append(arm)
        policy.update(arm, 0.0)
    assert sorted(pulled) == [0, 1, 2, 3]


def test_compute_ucb_values_shape_and_bonus():
    policy = UCB1(n_arms=3, seed=0)
    for arm in range(3):
        policy.update(arm, 0.5)
    ucb = policy._compute_ucb_values()
    assert ucb.shape == (3,)
    # bonus is positive, so UCB exceeds the raw value estimate
    assert np.all(ucb >= policy.values)


def test_prefers_higher_value_when_counts_equal():
    policy = UCB1(n_arms=3, seed=0)
    # equal counts, arm 2 has the highest empirical value
    policy.counts = np.array([10.0, 10.0, 10.0])
    policy.values = np.array([0.1, 0.2, 0.9])
    policy.total_pulls = 30
    assert policy.select_arm() == 2


def test_larger_exploration_factor_increases_bonus():
    low = UCB1(n_arms=2, exploration_factor=1.0, seed=0)
    high = UCB1(n_arms=2, exploration_factor=8.0, seed=0)
    for p in (low, high):
        p.counts = np.array([5.0, 5.0])
        p.values = np.array([0.5, 0.5])
        p.total_pulls = 10
    assert np.all(high._compute_ucb_values() > low._compute_ucb_values())
