"""
RovingBandit: A flexible library for multi-armed bandit algorithms.

Supports multiple objectives (regret minimization, best-arm identification,
variance minimization) in both online and batched modes.
"""

from rovingbandit.banditry import (
    arm_sequence,
    best_arm,
    pick_arm,
    pull_sequence,
    rep_bandit_cost,
    rep_bandit_rake,
    sim_runner,
)
from rovingbandit.core import (
    BanditEnvironment,
    History,
    InvalidArmError,
    InvalidConfigurationError,
    MissingConfigurationError,
    Objective,
    Policy,
    Result,
    RovingBanditError,
    UninitializedStateError,
)
from rovingbandit.objectives import (
    BestArmIdentification,
    RegretMinimization,
    VarianceMinimization,
)
from rovingbandit.policies import (
    LUCB,
    UCB1,
    BudgetedThompsonSampling,
    BudgetedUCB,
    EpsilonGreedy,
    EpsilonNeymanAllocation,
    ExploreFirst,
    GreedyPolicy,
    KasySautmann,
    LinUCB,
    RandomPolicy,
    RepresentationBandit,
    ThompsonSampling,
    TopTwoThompson,
)
from rovingbandit.runners import (
    BatchedRunner,
    OnlineRunner,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BanditEnvironment",
    "Policy",
    "Objective",
    "Result",
    "History",
    # Exceptions
    "RovingBanditError",
    "InvalidArmError",
    "InvalidConfigurationError",
    "MissingConfigurationError",
    "UninitializedStateError",
    # Policies
    "RandomPolicy",
    "GreedyPolicy",
    "EpsilonGreedy",
    "ExploreFirst",
    "UCB1",
    "BudgetedUCB",
    "ThompsonSampling",
    "BudgetedThompsonSampling",
    "RepresentationBandit",
    "EpsilonNeymanAllocation",
    "LUCB",
    "KasySautmann",
    "LinUCB",
    "TopTwoThompson",
    # Objectives
    "RegretMinimization",
    "BestArmIdentification",
    "VarianceMinimization",
    # Runners
    "OnlineRunner",
    "BatchedRunner",
    # Legacy functions (backward compatibility)
    "pick_arm",
    "sim_runner",
    "arm_sequence",
    "pull_sequence",
    "best_arm",
    "rep_bandit_cost",
    "rep_bandit_rake",
]
