"""Tests for the RegretMinimization objective."""

import numpy as np
import pytest

from rovingbandit import History, MissingConfigurationError, RandomPolicy, RegretMinimization


def _history(rewards):
    h = History()
    for r in rewards:
        h.add(0, r)
    return h


def test_compute_metric_uses_init_optimal():
    obj = RegretMinimization(optimal_reward=1.0)
    regret = obj.compute_metric(_history([1.0, 0.0, 1.0]))
    # optimal - reward summed: (0 + 1 + 0) = 1
    assert np.isclose(regret, 1.0)


def test_compute_metric_override_optimal():
    obj = RegretMinimization(optimal_reward=0.5)
    regret = obj.compute_metric(_history([0.0, 0.0]), optimal_reward=1.0)
    assert np.isclose(regret, 2.0)


def test_missing_optimal_raises():
    obj = RegretMinimization()
    with pytest.raises(MissingConfigurationError, match="optimal_reward"):
        obj.compute_metric(_history([1.0]))


def test_no_early_stopping():
    obj = RegretMinimization(optimal_reward=1.0)
    policy = RandomPolicy(n_arms=2, seed=0)
    assert obj.stopping_criterion(policy, _history([1.0])) is False


def test_metadata_includes_optimal_reward():
    obj = RegretMinimization(optimal_reward=0.7)
    policy = RandomPolicy(n_arms=2, seed=0)
    meta = obj.get_metadata(policy, _history([1.0]))
    assert meta == {"optimal_reward": 0.7}


def test_metadata_empty_without_optimal():
    obj = RegretMinimization()
    policy = RandomPolicy(n_arms=2, seed=0)
    assert obj.get_metadata(policy, _history([1.0])) == {}
