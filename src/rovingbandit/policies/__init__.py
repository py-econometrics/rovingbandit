"""Policy implementations for bandit algorithms.

Policies are organized into subpackages by the objective they target:
``regret_minimization``, ``best_arm_identification``, and ``variance_minimization``.
All public classes are re-exported here for backward-compatible imports.
"""

from rovingbandit.policies.best_arm_identification import LUCB, TopTwoThompson
from rovingbandit.policies.regret_minimization import (
    UCB1,
    BudgetedThompsonSampling,
    BudgetedUCB,
    EpsilonGreedy,
    ExploreFirst,
    GreedyPolicy,
    LinUCB,
    RandomPolicy,
    ThompsonSampling,
)
from rovingbandit.policies.variance_minimization import (
    EpsilonNeymanAllocation,
    KasySautmann,
    RepresentationBandit,
)

__all__ = [
    "RandomPolicy",
    "GreedyPolicy",
    "EpsilonGreedy",
    "ExploreFirst",
    "UCB1",
    "BudgetedUCB",
    "ThompsonSampling",
    "BudgetedThompsonSampling",
    "TopTwoThompson",
    "RepresentationBandit",
    "EpsilonNeymanAllocation",
    "LUCB",
    "KasySautmann",
    "LinUCB",
]
