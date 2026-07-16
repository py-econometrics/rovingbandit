"""Backward-compatibility tests for the pre-reorg flat policy module paths.

The policies were moved into objective subpackages (regret_minimization,
best_arm_identification, variance_minimization); shim modules at the old flat
paths must keep deep imports like ``from rovingbandit.policies.ucb import UCB1``
working.
"""

import importlib

import pytest

import rovingbandit.policies

# (old flat module name, public class name) for every pre-reorg module.
OLD_MODULE_CLASSES = [
    ("budgeted_thompson", "BudgetedThompsonSampling"),
    ("budgeted_ucb", "BudgetedUCB"),
    ("epsilon_greedy", "EpsilonGreedy"),
    ("epsilon_neyman", "EpsilonNeymanAllocation"),
    ("explore_first", "ExploreFirst"),
    ("greedy", "GreedyPolicy"),
    ("kasy_sautmann", "KasySautmann"),
    ("linucb", "LinUCB"),
    ("lucb", "LUCB"),
    ("random_policy", "RandomPolicy"),
    ("representation_bandit", "RepresentationBandit"),
    ("thompson_sampling", "ThompsonSampling"),
    ("top_two_thompson", "TopTwoThompson"),
    ("ucb", "UCB1"),
]


@pytest.mark.parametrize(("module_name", "class_name"), OLD_MODULE_CLASSES)
def test_old_module_path_exposes_class(module_name, class_name):
    module = importlib.import_module(f"rovingbandit.policies.{module_name}")
    assert getattr(module, class_name) is getattr(rovingbandit.policies, class_name)


def test_representative_deep_imports():
    from rovingbandit.policies.random_policy import RandomPolicy
    from rovingbandit.policies.ucb import UCB1

    assert UCB1 is rovingbandit.policies.UCB1
    assert RandomPolicy is rovingbandit.policies.RandomPolicy
