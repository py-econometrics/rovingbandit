"""Tests for EpsilonNeymanAllocation."""

import numpy as np
import pytest

from rovingbandit import (
    EpsilonNeymanAllocation,
    InvalidConfigurationError,
    MissingConfigurationError,
)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_invalid_exploration_fraction_raises(fraction):
    with pytest.raises(InvalidConfigurationError, match="exploration_fraction"):
        EpsilonNeymanAllocation(n_arms=3, exploration_fraction=fraction)


def test_select_arm_requires_horizon():
    policy = EpsilonNeymanAllocation(n_arms=3, seed=0)
    with pytest.raises(MissingConfigurationError, match="horizon"):
        policy.select_arm()


def test_exploration_phase_returns_valid_arm():
    policy = EpsilonNeymanAllocation(n_arms=3, exploration_fraction=0.2, horizon=50, seed=42)
    assert 0 <= policy.select_arm() < 3


def test_allocation_favors_highest_variance_arm():
    policy = EpsilonNeymanAllocation(n_arms=3, exploration_fraction=0.2, horizon=50, seed=42)
    policy.counts = np.array([10.0, 10.0, 10.0])
    policy.values = np.array([0.1, 0.5, 0.9])
    policy.total_pulls = 30
    probs = policy._allocation_probabilities()
    assert probs.shape == (3,)
    assert np.isclose(np.sum(probs), 1.0)
    # variance peaks at p=0.5 -> arm 1 gets the largest allocation
    assert probs[1] > probs[0]
    assert probs[1] > probs[2]


def test_unpulled_arms_get_exploration_weight():
    policy = EpsilonNeymanAllocation(n_arms=3, exploration_fraction=0.2, horizon=50, seed=0)
    policy.counts = np.array([5.0, 0.0, 5.0])
    policy.values = np.array([0.5, 0.0, 0.5])
    policy.total_pulls = 10
    probs = policy._allocation_probabilities()
    # the unpulled arm receives the safeguard sigma (0.5), so nonzero mass
    assert probs[1] > 0


def test_get_allocation_probabilities_after_compute():
    policy = EpsilonNeymanAllocation(n_arms=2, exploration_fraction=0.2, horizon=50, seed=0)
    assert policy.get_allocation_probabilities() is None
    policy.counts = np.array([4.0, 4.0])
    policy.values = np.array([0.5, 0.5])
    policy.total_pulls = 8
    probs = policy._allocation_probabilities()
    assert np.array_equal(policy.get_allocation_probabilities(), probs)


def test_state_roundtrip_includes_exploration_settings():
    policy = EpsilonNeymanAllocation(
        n_arms=3, exploration_fraction=0.25, horizon=80, min_variance=1e-5, seed=0
    )
    policy.counts = np.array([4.0, 4.0, 4.0])
    policy.values = np.array([0.1, 0.5, 0.9])
    policy.total_pulls = 12
    policy._allocation_probabilities()
    state = policy.get_state()

    restored = EpsilonNeymanAllocation(n_arms=3, seed=1)
    restored.set_state(state)
    assert restored.exploration_fraction == 0.25
    assert restored.horizon == 80
    assert restored.min_variance == 1e-5
    assert np.array_equal(
        restored.get_allocation_probabilities(), policy.get_allocation_probabilities()
    )
