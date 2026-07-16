"""Tests for RandomPolicy."""

import numpy as np

from rovingbandit import RandomPolicy


def test_select_arm_in_range():
    policy = RandomPolicy(n_arms=5, seed=42)
    for _ in range(50):
        assert 0 <= policy.select_arm() < 5


def test_update_records_reward():
    policy = RandomPolicy(n_arms=5, seed=42)
    arm = policy.select_arm()
    policy.update(arm, 1.0)
    assert policy.counts[arm] == 1
    assert policy.values[arm] == 1.0


def test_selection_is_roughly_uniform():
    policy = RandomPolicy(n_arms=4, seed=0)
    counts = np.zeros(4)
    for _ in range(4000):
        counts[policy.select_arm()] += 1
    shares = counts / counts.sum()
    assert np.all(np.abs(shares - 0.25) < 0.05)
