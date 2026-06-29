"""Tests for KasySautmann."""

import numpy as np

from rovingbandit import KasySautmann


def test_unpulled_safeguard_is_uniform_over_unpulled():
    policy = KasySautmann(n_arms=3, seed=1)
    policy.counts = np.array([2.0, 0.0, 0.0])
    probs = policy._allocation_probabilities()
    assert probs[0] == 0.0
    assert np.isclose(probs[1], 0.5)
    assert np.isclose(probs[2], 0.5)


def test_welfare_constraint_suppresses_low_arms():
    policy = KasySautmann(n_arms=2, welfare_threshold=0.75, smoothing=1e-4, seed=1)
    # arm 0 below the welfare floor (0.75 * 0.9), arm 1 above
    policy.values = np.array([0.2, 0.9])
    policy.counts = np.array([20.0, 20.0])
    policy.total_pulls = 40
    probs = policy._allocation_probabilities()
    assert probs[1] > probs[0]
    assert np.isclose(np.sum(probs), 1.0)


def test_smoothing_keeps_probabilities_positive():
    policy = KasySautmann(n_arms=3, welfare_threshold=0.8, smoothing=1e-2, seed=0)
    # all means equal -> sigmas equal; smoothing keeps every prob > 0
    policy.values = np.array([0.5, 0.5, 0.5])
    policy.counts = np.array([10.0, 10.0, 10.0])
    policy.total_pulls = 30
    probs = policy._allocation_probabilities()
    assert np.all(probs > 0)
    assert np.isclose(np.sum(probs), 1.0)


def test_select_arm_returns_valid_index():
    policy = KasySautmann(n_arms=2, seed=1)
    policy.values = np.array([0.2, 0.9])
    policy.counts = np.array([20.0, 20.0])
    policy.total_pulls = 40
    assert all(policy.select_arm() in (0, 1) for _ in range(10))


def test_update_clears_cached_probabilities():
    policy = KasySautmann(n_arms=2, seed=0)
    policy.counts = np.array([5.0, 5.0])
    policy.values = np.array([0.5, 0.5])
    policy.total_pulls = 10
    policy._allocation_probabilities()
    assert policy.get_allocation_probabilities() is not None
    policy.update(0, 1.0)
    assert policy.get_allocation_probabilities() is None


def test_state_roundtrip_includes_welfare_settings():
    policy = KasySautmann(n_arms=3, welfare_threshold=0.6, smoothing=1e-2, seed=0)
    policy.counts = np.array([5.0, 5.0, 5.0])
    policy.values = np.array([0.2, 0.5, 0.9])
    policy.total_pulls = 15
    policy._allocation_probabilities()
    state = policy.get_state()

    restored = KasySautmann(n_arms=3, seed=1)
    restored.set_state(state)
    assert restored.welfare_threshold == 0.6
    assert restored.smoothing == 1e-2
    assert np.array_equal(
        restored.get_allocation_probabilities(), policy.get_allocation_probabilities()
    )


def test_reset_clears_state():
    policy = KasySautmann(n_arms=2, seed=0)
    policy.counts = np.array([5.0, 5.0])
    policy.values = np.array([0.5, 0.5])
    policy.total_pulls = 10
    policy._allocation_probabilities()
    policy.reset()
    assert policy.total_pulls == 0
    assert policy.get_allocation_probabilities() is None
