"""Tests for the Objective abstract base class contract."""

import pytest

from rovingbandit import History, Objective, RandomPolicy


class _MinimalObjective(Objective):
    """Concrete objective that only implements the abstract methods."""

    def compute_metric(self, history, **kwargs):
        return 0.0

    def stopping_criterion(self, policy, history, **kwargs):
        return False


def test_cannot_instantiate_abstract_objective():
    with pytest.raises(TypeError):
        Objective()  # type: ignore[abstract]


def test_default_get_metadata_is_empty():
    """A subclass that does not override get_metadata gets the default empty dict."""
    obj = _MinimalObjective()
    policy = RandomPolicy(n_arms=2, seed=0)
    history = History()
    assert obj.get_metadata(policy, history) == {}


def test_minimal_subclass_methods_callable():
    obj = _MinimalObjective()
    policy = RandomPolicy(n_arms=2, seed=0)
    history = History()
    assert obj.compute_metric(history) == 0.0
    assert obj.stopping_criterion(policy, history) is False
