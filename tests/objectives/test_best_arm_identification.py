"""Tests for the BestArmIdentification objective."""

import numpy as np
import pytest

from rovingbandit import (
    UCB1,
    BestArmIdentification,
    History,
    MissingConfigurationError,
    ThompsonSampling,
)


def test_compute_metric_requires_policy():
    obj = BestArmIdentification(seed=0)
    with pytest.raises(MissingConfigurationError, match="policy required"):
        obj.compute_metric(History(), policy=None)


def test_confidence_in_unit_interval():
    obj = BestArmIdentification(n_mc_samples=200, seed=0)
    policy = ThompsonSampling(n_arms=3, seed=0)
    policy.successes = np.array([50.0, 5.0, 5.0])
    policy.failures = np.array([5.0, 50.0, 50.0])
    conf = obj.compute_metric(History(), policy=policy)
    assert 0.0 <= conf <= 1.0
    # arm 0 clearly dominates -> high confidence
    assert conf > 0.8


def test_thompson_style_vs_empirical_fallback_agree_on_clear_winner():
    # Thompson-style policy exposes successes/failures
    ts = ThompsonSampling(n_arms=2, seed=0)
    ts.successes = np.array([90.0, 10.0])
    ts.failures = np.array([10.0, 90.0])

    # UCB1 lacks successes/failures -> exercises the empirical-mean fallback branch
    ucb = UCB1(n_arms=2, seed=0)
    ucb.values = np.array([0.9, 0.1])
    ucb.counts = np.array([100.0, 100.0])

    obj = BestArmIdentification(n_mc_samples=500, seed=0)
    assert obj.compute_metric(History(), policy=ts) > 0.8
    assert obj.compute_metric(History(), policy=ucb) > 0.8


def test_stopping_criterion_threshold():
    obj = BestArmIdentification(confidence_threshold=0.8, n_mc_samples=500, seed=0)
    policy = ThompsonSampling(n_arms=2, seed=0)
    policy.successes = np.array([200.0, 1.0])
    policy.failures = np.array([1.0, 200.0])
    assert obj.stopping_criterion(policy, History()) is True


def test_metadata_reports_confidence_and_best_arm():
    obj = BestArmIdentification(n_mc_samples=200, seed=0)
    policy = ThompsonSampling(n_arms=3, seed=0)
    policy.successes = np.array([5.0, 50.0, 5.0])
    policy.failures = np.array([50.0, 5.0, 50.0])
    policy.values = np.array([0.1, 0.9, 0.1])
    meta = obj.get_metadata(policy, History())
    assert "confidence" in meta
    assert meta["best_arm"] == 1


def test_seeded_reproducibility():
    policy = ThompsonSampling(n_arms=3, seed=0)
    policy.successes = np.array([20.0, 10.0, 5.0])
    policy.failures = np.array([5.0, 10.0, 20.0])
    a = BestArmIdentification(n_mc_samples=300, seed=123).compute_metric(History(), policy=policy)
    b = BestArmIdentification(n_mc_samples=300, seed=123).compute_metric(History(), policy=policy)
    assert a == b
