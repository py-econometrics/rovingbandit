"""Tests for the VarianceMinimization objective."""

import numpy as np
import pytest

from rovingbandit import (
    EpsilonNeymanAllocation,
    History,
    MissingConfigurationError,
    RandomPolicy,
    VarianceMinimization,
)


def test_compute_metric_requires_policy():
    obj = VarianceMinimization()
    with pytest.raises(MissingConfigurationError, match="policy required"):
        obj.compute_metric(History(), policy=None)


def test_variance_sum_with_zero_count_clamp():
    obj = VarianceMinimization()
    policy = RandomPolicy(n_arms=2, seed=0)
    # arm 0 pulled (variance 0.25 / 4), arm 1 unpulled (count clamped to 1)
    policy.values = np.array([0.5, 0.5])
    policy.counts = np.array([4.0, 0.0])
    variance = obj.compute_metric(History(), policy=policy)
    # 0.25/4 + 0.25/1 = 0.0625 + 0.25
    assert np.isclose(variance, 0.0625 + 0.25)


def test_no_early_stopping():
    obj = VarianceMinimization()
    policy = RandomPolicy(n_arms=2, seed=0)
    assert obj.stopping_criterion(policy, History()) is False


def test_metadata_reports_variance():
    obj = VarianceMinimization()
    policy = RandomPolicy(n_arms=2, seed=0)
    policy.values = np.array([0.5, 0.5])
    policy.counts = np.array([2.0, 2.0])
    meta = obj.get_metadata(policy, History())
    assert "estimation_variance" in meta


def test_metadata_group_shares_from_arm_groups():
    obj = VarianceMinimization()
    policy = RandomPolicy(n_arms=2, seed=0)
    policy.counts = np.array([1.0, 1.0])
    history = History()
    for arm in [0, 0, 1]:  # group 0 pulled twice, group 1 once
        history.add(arm, 1.0)
    arm_groups = np.array([0, 1])
    meta = obj.get_metadata(policy, history, arm_groups=arm_groups)
    assert "group_shares" in meta
    assert np.allclose(meta["group_shares"], np.array([2 / 3, 1 / 3]))


def test_metadata_includes_allocation_probabilities():
    obj = VarianceMinimization()
    policy = EpsilonNeymanAllocation(n_arms=3, horizon=50, seed=0)
    policy.counts = np.array([4.0, 4.0, 4.0])
    policy.values = np.array([0.1, 0.5, 0.9])
    policy.total_pulls = 12
    policy._allocation_probabilities()
    meta = obj.get_metadata(policy, History())
    assert "allocation_probabilities" in meta
    assert np.isclose(np.sum(meta["allocation_probabilities"]), 1.0)
