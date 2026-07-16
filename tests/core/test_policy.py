"""Tests for the Policy base class behavior, exercised via concrete subclasses."""

import numpy as np
import pytest

from rovingbandit import GreedyPolicy, RandomPolicy


class TestStateManagement:
    """reset / get_state / set_state."""

    def test_initial_state_is_zeroed(self):
        policy = RandomPolicy(n_arms=4, seed=0)
        assert policy.total_pulls == 0
        assert np.allclose(policy.counts, np.zeros(4))
        assert np.allclose(policy.values, np.zeros(4))

    def test_reset_zeroes_state(self):
        policy = RandomPolicy(n_arms=3, seed=0)
        for _ in range(5):
            policy.update(0, 1.0)
        assert policy.total_pulls == 5
        policy.reset()
        assert policy.total_pulls == 0
        assert np.allclose(policy.counts, np.zeros(3))
        assert np.allclose(policy.values, np.zeros(3))

    def test_get_set_state_roundtrip(self):
        policy = GreedyPolicy(n_arms=3, seed=1)
        for arm, reward in [(0, 1.0), (1, 0.5), (1, 0.7), (2, 0.2)]:
            policy.update(arm, reward)
        state = policy.get_state()

        restored = GreedyPolicy(n_arms=3, seed=2)
        restored.set_state(state)
        assert restored.n_arms == policy.n_arms
        assert restored.total_pulls == policy.total_pulls
        assert np.array_equal(restored.counts, policy.counts)
        assert np.array_equal(restored.values, policy.values)

    def test_get_state_returns_copies(self):
        policy = RandomPolicy(n_arms=2, seed=0)
        policy.update(0, 1.0)
        state = policy.get_state()
        state["counts"][0] = 999.0
        assert policy.counts[0] == 1.0  # internal state untouched


class TestIncrementalUpdate:
    """Running-mean value updates."""

    def test_running_mean(self):
        policy = RandomPolicy(n_arms=1, seed=0)
        rewards = [1.0, 0.0, 1.0, 1.0]
        for r in rewards:
            policy.update(0, r)
        assert policy.counts[0] == 4
        assert np.isclose(policy.values[0], np.mean(rewards))

    def test_update_tracks_counts_and_total(self):
        policy = RandomPolicy(n_arms=3, seed=0)
        policy.update(0, 1.0)
        policy.update(0, 0.0)
        policy.update(2, 1.0)
        assert policy.counts[0] == 2
        assert policy.counts[1] == 0
        assert policy.counts[2] == 1
        assert policy.total_pulls == 3


class TestGreedySelection:
    """Greedy arm selection and tie-breaking."""

    def test_selects_highest_value(self):
        policy = GreedyPolicy(n_arms=4, seed=0)
        for arm, reward in [(0, 0.1), (1, 0.9), (2, 0.3), (3, 0.2)]:
            policy.update(arm, reward)
        assert policy.select_arm() == 1

    def test_tie_break_within_argmax_set(self):
        policy = GreedyPolicy(n_arms=4, seed=0)
        # arms 1 and 3 tie for the max value
        for arm, reward in [(0, 0.1), (1, 0.9), (2, 0.3), (3, 0.9)]:
            policy.update(arm, reward)
        choices = {policy.select_arm() for _ in range(50)}
        assert choices <= {1, 3}

    def test_tie_break_is_seed_deterministic(self):
        kwargs = {"n_arms": 4, "seed": 123}
        a = GreedyPolicy(**kwargs)
        b = GreedyPolicy(**kwargs)
        for arm, reward in [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]:
            a.update(arm, reward)
            b.update(arm, reward)
        assert [a.select_arm() for _ in range(20)] == [b.select_arm() for _ in range(20)]


def test_cannot_instantiate_abstract_policy():
    """The abstract base class cannot be instantiated directly."""
    from rovingbandit import Policy

    with pytest.raises(TypeError):
        Policy(n_arms=3)  # type: ignore[abstract]
