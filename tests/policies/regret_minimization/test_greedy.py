"""Tests for GreedyPolicy."""

from rovingbandit import GreedyPolicy


def test_first_pull_without_data_is_valid():
    policy = GreedyPolicy(n_arms=5, seed=42)
    # total_pulls == 0 -> random but in range
    assert 0 <= policy.select_arm() < 5


def test_exploits_best_arm():
    policy = GreedyPolicy(n_arms=5, seed=42)
    for i in range(5):
        policy.update(i, i * 0.1)
    assert policy.select_arm() == 4


def test_greedy_never_explores_after_data():
    policy = GreedyPolicy(n_arms=3, seed=1)
    policy.update(0, 0.0)
    policy.update(1, 1.0)
    policy.update(2, 0.5)
    assert all(policy.select_arm() == 1 for _ in range(20))
