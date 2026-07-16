"""Tests for EpsilonGreedy."""

import numpy as np

from rovingbandit import EpsilonGreedy


def test_epsilon_zero_is_pure_greedy():
    policy = EpsilonGreedy(n_arms=4, epsilon=0.0, seed=42)
    for arm, reward in [(0, 0.1), (1, 0.9), (2, 0.2), (3, 0.3)]:
        policy.update(arm, reward)
    assert all(policy.select_arm() == 1 for _ in range(20))


def test_epsilon_one_is_pure_random():
    policy = EpsilonGreedy(n_arms=4, epsilon=1.0, seed=0)
    for arm, reward in [(0, 0.1), (1, 0.9), (2, 0.2), (3, 0.3)]:
        policy.update(arm, reward)
    chosen = {policy.select_arm() for _ in range(100)}
    # explores across multiple arms rather than locking onto the greedy one
    assert len(chosen) > 1


def test_decay_shrinks_epsilon_over_time():
    policy = EpsilonGreedy(n_arms=3, epsilon=0.5, decay=True, seed=0)
    assert policy.epsilon == 0.5
    for _ in range(100):
        arm = policy.select_arm()
        policy.update(arm, 1.0)
    # epsilon is recomputed at select time as epsilon_0 / sqrt(total_pulls)
    assert policy.epsilon < 0.5
    policy.select_arm()  # recompute with total_pulls == 100
    assert np.isclose(policy.epsilon, 0.5 / np.sqrt(100))


def test_no_decay_keeps_epsilon_constant():
    policy = EpsilonGreedy(n_arms=3, epsilon=0.3, decay=False, seed=0)
    for _ in range(50):
        arm = policy.select_arm()
        policy.update(arm, 1.0)
    assert policy.epsilon == 0.3
