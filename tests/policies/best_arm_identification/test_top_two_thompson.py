"""Tests for TopTwoThompson."""

import numpy as np
import pytest

from rovingbandit import InvalidConfigurationError, TopTwoThompson


@pytest.mark.parametrize("psi", [-0.1, 1.5, 2.0])
def test_invalid_psi_raises(psi):
    with pytest.raises(InvalidConfigurationError, match="psi must be between 0 and 1"):
        TopTwoThompson(n_arms=3, psi=psi)


def test_psi_one_always_plays_leader():
    # psi=1 -> always returns the sampled leader (never resamples a challenger)
    policy = TopTwoThompson(n_arms=3, psi=1.0, seed=0)
    policy.successes = np.array([100.0, 1.0, 1.0])
    policy.failures = np.array([1.0, 100.0, 100.0])
    # arm 0 dominates the posterior, so the leader is almost always arm 0
    counts = np.zeros(3)
    for _ in range(100):
        counts[policy.select_arm()] += 1
    assert counts[0] > 90


def test_psi_zero_picks_challenger():
    # psi=0 -> always look for a challenger different from the leader
    policy = TopTwoThompson(n_arms=2, psi=0.0, seed=0)
    policy.successes = np.array([100.0, 1.0])
    policy.failures = np.array([1.0, 100.0])
    # leader is arm 0; challenger must be arm 1
    assert all(policy.select_arm() == 1 for _ in range(20))


def test_max_resamples_fallback_when_concentrated():
    # Force the resample loop to exhaust by making max_resamples tiny and the
    # posterior extremely concentrated on the leader.
    policy = TopTwoThompson(n_arms=3, psi=0.0, max_resamples=1, seed=0)
    policy.successes = np.array([1e6, 0.0, 0.0])
    policy.failures = np.array([0.0, 1e6, 1e6])
    arm = policy.select_arm()
    # fallback masks the leader (arm 0) and returns a distinct arm
    assert arm != 0
    assert arm in (1, 2)


def test_update_tracks_posteriors():
    policy = TopTwoThompson(n_arms=3, seed=0)
    arm = policy.select_arm()
    policy.update(arm, 1.0)
    assert policy.counts[arm] == 1
    assert policy.successes[arm] == 1.0


def test_state_roundtrip_includes_ttts_params():
    policy = TopTwoThompson(n_arms=4, psi=0.3, max_resamples=50, seed=1)
    policy.update(0, 1.0)
    state = policy.get_state()
    assert state["psi"] == 0.3
    assert state["max_resamples"] == 50

    restored = TopTwoThompson(n_arms=4)
    restored.set_state(state)
    assert restored.psi == 0.3
    assert restored.max_resamples == 50
    assert np.array_equal(restored.successes, policy.successes)
