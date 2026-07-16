"""Tests for RepresentationBandit."""

import numpy as np
import pytest

from rovingbandit import InvalidConfigurationError, RepresentationBandit


def _make(**overrides):
    kwargs = {
        "n_arms": 2,
        "arm_groups": np.array([0, 1]),
        "target_shares": np.array([0.2, 0.8]),
        "total_budget": 100.0,
        "costs": np.array([1.0, 1.0]),
        "seed": 42,
    }
    kwargs.update(overrides)
    return RepresentationBandit(**kwargs)


def test_arm_groups_length_mismatch_raises():
    with pytest.raises(InvalidConfigurationError, match="arm_groups length"):
        _make(arm_groups=np.array([0, 1, 0]))


def test_target_shares_must_sum_to_one():
    with pytest.raises(InvalidConfigurationError, match="sum to 1.0"):
        _make(target_shares=np.array([0.3, 0.3]))


def test_arm_groups_indices_must_be_within_range():
    with pytest.raises(InvalidConfigurationError, match="exceed number of target_shares"):
        _make(arm_groups=np.array([0, 5]))


def test_favors_under_represented_group():
    policy = _make()
    # equal 50/50 pulls over-represents group 0 (target 0.2) and under-represents group 1
    policy.counts = np.array([50.0, 50.0])
    policy.total_pulls = 100
    policy.total_cost_incurred = 50.0
    policy.successes[:] = 10
    policy.failures[:] = 10
    # group 1's effective cost is reduced -> arm 1 is preferred
    assert policy.select_arm() == 1


def test_update_accumulates_total_cost():
    policy = _make()
    policy.update(0, 1.0, cost=2.0)
    policy.update(1, 0.0, cost=3.0)
    assert policy.total_cost_incurred == 5.0


def test_state_roundtrip_includes_representation_params():
    policy = _make()
    policy.update(0, 1.0, cost=4.0)
    state = policy.get_state()
    restored = _make()
    restored.set_state(state)
    assert restored.total_cost_incurred == 4.0
    assert np.array_equal(restored.target_shares, policy.target_shares)
    assert restored.total_budget == policy.total_budget
