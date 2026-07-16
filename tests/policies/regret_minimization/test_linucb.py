"""Tests for LinUCB (linear contextual bandit)."""

import numpy as np
import pytest

from rovingbandit import (
    InvalidConfigurationError,
    LinUCB,
    MissingConfigurationError,
    OnlineRunner,
)


def test_select_arm_requires_context():
    policy = LinUCB(n_arms=2, seed=0)
    with pytest.raises(MissingConfigurationError, match="context"):
        policy.select_arm(context=None)


def test_context_dimension_mismatch_raises():
    policy = LinUCB(n_arms=2, seed=0)
    bad_context = np.ones((3, 4))  # first dim != n_arms
    with pytest.raises(InvalidConfigurationError, match="first dimension"):
        policy.select_arm(context=bad_context)


def test_update_without_context_raises():
    policy = LinUCB(n_arms=2, seed=0)
    with pytest.raises(MissingConfigurationError, match="context required"):
        policy.update(0, 1.0)


def test_matrices_initialized_on_first_selection():
    policy = LinUCB(n_arms=2, seed=0)
    assert policy._A is None
    context = np.array([[1.0, 0.0], [0.0, 1.0]])
    policy.select_arm(context=context)
    assert policy._A is not None
    assert policy._A.shape == (2, 2, 2)


def test_recovers_linear_reward_ordering(contextual_env):
    policy = LinUCB(n_arms=2, alpha=0.1, seed=0)
    runner = OnlineRunner()
    result = runner.run(policy, contextual_env, n_steps=60)
    counts = result.policy_state["counts"]
    values = result.policy_state["values"]
    # arm 0 has the higher expected reward under the fixed contexts
    assert counts[0] > counts[1]
    assert values[0] > values[1]
