"""Tests for LUCB."""

import numpy as np

from rovingbandit import LUCB


def test_initial_phase_pulls_each_arm_once():
    policy = LUCB(n_arms=3, seed=0)
    pulled = []
    for _ in range(3):
        arm = policy.select_arm()
        pulled.append(arm)
        policy.update(arm, 0.0)
    assert set(pulled) == {0, 1, 2}


def test_confidence_radius_shrinks_with_pulls():
    policy = LUCB(n_arms=2, seed=0)
    policy.counts = np.array([5.0, 100.0])
    policy.total_pulls = 105
    radii = policy._confidence_radius()
    # the more-pulled arm has the smaller radius
    assert radii[1] < radii[0]


def test_focuses_on_leader_or_challenger():
    policy = LUCB(n_arms=3, exploration_factor=2.0, seed=0)
    # arm 2 leads; arm 1 is uncertain (few pulls) -> challenger
    policy.values = np.array([0.2, 0.6, 0.8])
    policy.counts = np.array([50.0, 5.0, 50.0])
    policy.total_pulls = 105
    assert policy.select_arm() in (1, 2)


def test_picks_more_uncertain_of_the_pair():
    policy = LUCB(n_arms=2, exploration_factor=2.0, seed=0)
    # leader is arm 0; challenger arm 1 has far fewer pulls -> larger radius
    policy.values = np.array([0.9, 0.4])
    policy.counts = np.array([100.0, 2.0])
    policy.total_pulls = 102
    # challenger is more uncertain, so LUCB samples it
    assert policy.select_arm() == 1
