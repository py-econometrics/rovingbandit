"""Tests for ThompsonSampling."""

import numpy as np

from rovingbandit import ThompsonSampling


def test_update_tracks_successes_and_failures():
    policy = ThompsonSampling(n_arms=3, seed=42)
    policy.update(0, 1.0)
    policy.update(0, 0.0)
    policy.update(1, 1.0)
    assert policy.successes[0] == 1
    assert policy.failures[0] == 1
    assert policy.successes[1] == 1
    assert policy.total_pulls == 3


def test_sample_posteriors_shape_and_range():
    policy = ThompsonSampling(n_arms=4, seed=0)
    samples = policy._sample_posteriors()
    assert samples.shape == (4,)
    assert np.all((samples >= 0) & (samples <= 1))


def test_concentrated_posterior_favors_best_arm():
    policy = ThompsonSampling(n_arms=2, seed=0)
    policy.successes = np.array([90.0, 10.0])
    policy.failures = np.array([10.0, 90.0])
    counts = np.zeros(2)
    for _ in range(200):
        counts[policy.select_arm()] += 1
    assert counts[0] > counts[1]


def test_get_set_state_roundtrip_includes_posteriors():
    policy = ThompsonSampling(n_arms=3, prior_alpha=2.0, prior_beta=3.0, seed=1)
    for arm, reward in [(0, 1.0), (1, 0.0), (2, 1.0)]:
        policy.update(arm, reward)
    state = policy.get_state()
    restored = ThompsonSampling(n_arms=3, seed=2)
    restored.set_state(state)
    assert np.array_equal(restored.successes, policy.successes)
    assert np.array_equal(restored.failures, policy.failures)
    assert restored.prior_alpha == 2.0
    assert restored.prior_beta == 3.0


def test_reset_clears_posteriors():
    policy = ThompsonSampling(n_arms=2, seed=0)
    policy.update(0, 1.0)
    policy.update(1, 0.0)
    policy.reset()
    assert np.allclose(policy.successes, 0.0)
    assert np.allclose(policy.failures, 0.0)
    assert policy.total_pulls == 0


def test_seed_determinism():
    a = ThompsonSampling(n_arms=3, seed=99)
    b = ThompsonSampling(n_arms=3, seed=99)
    assert [a.select_arm() for _ in range(10)] == [b.select_arm() for _ in range(10)]
