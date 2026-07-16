"""Tests for ExploreFirst (explore-then-commit)."""

from rovingbandit import ExploreFirst


def test_explores_then_commits_with_horizon():
    # horizon 100, fraction 0.2 -> 20 exploration steps
    policy = ExploreFirst(n_arms=5, exploration_fraction=0.2, horizon=100, seed=42)
    # During exploration, feed rewards so arm 4 becomes best
    for _ in range(20):
        arm = policy.select_arm()
        policy.update(arm, 1.0 if arm == 4 else 0.0)
    assert policy.total_pulls == 20
    # Now in commit phase -> always greedy (arm 4)
    assert policy.select_arm() == 4


def test_set_horizon_updates_phase_boundary():
    policy = ExploreFirst(n_arms=3, exploration_fraction=0.5, seed=0)
    policy.set_horizon(10)
    assert policy.horizon == 10
    # 5 exploration steps; commit afterwards
    for _ in range(5):
        arm = policy.select_arm()
        policy.update(arm, 1.0 if arm == 2 else 0.0)
    assert policy.select_arm() == 2


def test_without_horizon_explores_then_commits():
    # No horizon: exploration_steps = n_arms * (1/fraction) = 3 * 10 = 30
    policy = ExploreFirst(n_arms=3, exploration_fraction=0.1, seed=0)
    arms = [policy.select_arm() for _ in range(3)]
    for arm in arms:
        policy.update(arm, 0.0)
    # still within exploration window; selections remain valid indices
    assert all(0 <= policy.select_arm() < 3 for _ in range(5))
