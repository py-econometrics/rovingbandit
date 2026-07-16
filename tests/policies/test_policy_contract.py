"""Contract tests shared across all policies that select without extra setup.

Each policy must: select a valid arm index, mutate counts/total_pulls on update,
and behave deterministically under a fixed seed.
"""

import numpy as np
import pytest

from rovingbandit import (
    LUCB,
    UCB1,
    BudgetedThompsonSampling,
    BudgetedUCB,
    EpsilonGreedy,
    ExploreFirst,
    GreedyPolicy,
    KasySautmann,
    RandomPolicy,
    ThompsonSampling,
    TopTwoThompson,
)

N_ARMS = 4
SEED = 7

# Builders for policies whose select_arm() works with no context / horizon setup.
POLICY_BUILDERS = {
    "RandomPolicy": lambda: RandomPolicy(n_arms=N_ARMS, seed=SEED),
    "GreedyPolicy": lambda: GreedyPolicy(n_arms=N_ARMS, seed=SEED),
    "EpsilonGreedy": lambda: EpsilonGreedy(n_arms=N_ARMS, epsilon=0.2, seed=SEED),
    "ExploreFirst": lambda: ExploreFirst(n_arms=N_ARMS, horizon=50, seed=SEED),
    "UCB1": lambda: UCB1(n_arms=N_ARMS, seed=SEED),
    "ThompsonSampling": lambda: ThompsonSampling(n_arms=N_ARMS, seed=SEED),
    "BudgetedUCB": lambda: BudgetedUCB(n_arms=N_ARMS, seed=SEED),
    "BudgetedThompsonSampling": lambda: BudgetedThompsonSampling(n_arms=N_ARMS, seed=SEED),
    "TopTwoThompson": lambda: TopTwoThompson(n_arms=N_ARMS, seed=SEED),
    "LUCB": lambda: LUCB(n_arms=N_ARMS, seed=SEED),
    "KasySautmann": lambda: KasySautmann(n_arms=N_ARMS, seed=SEED),
}

POLICY_NAMES = list(POLICY_BUILDERS)


@pytest.fixture(params=POLICY_NAMES)
def policy_builder(request):
    return POLICY_BUILDERS[request.param]


def test_select_arm_in_range(policy_builder):
    policy = policy_builder()
    for _ in range(20):
        arm = policy.select_arm()
        assert isinstance(arm, int)
        assert 0 <= arm < N_ARMS


def test_update_mutates_state(policy_builder):
    policy = policy_builder()
    arm = policy.select_arm()
    policy.update(arm, 1.0)
    assert policy.counts[arm] == 1
    assert policy.total_pulls == 1


def test_seed_determinism(policy_builder):
    a = policy_builder()
    b = policy_builder()
    seq_a, seq_b = [], []
    for _ in range(15):
        arm_a = a.select_arm()
        arm_b = b.select_arm()
        seq_a.append(arm_a)
        seq_b.append(arm_b)
        reward = 1.0 if arm_a % 2 == 0 else 0.0
        a.update(arm_a, reward)
        b.update(arm_b, reward)
    assert seq_a == seq_b


def test_full_run_distributes_pulls(policy_builder):
    policy = policy_builder()
    for _ in range(40):
        arm = policy.select_arm()
        policy.update(arm, float(np.random.default_rng(arm).random()))
    assert policy.total_pulls == 40
    assert policy.counts.sum() == 40
