"""Policies aimed at variance minimization and representation targeting."""

from rovingbandit.policies.variance_minimization.epsilon_neyman import (
    EpsilonNeymanAllocation,
)
from rovingbandit.policies.variance_minimization.kasy_sautmann import KasySautmann
from rovingbandit.policies.variance_minimization.representation_bandit import (
    RepresentationBandit,
)

__all__ = [
    "EpsilonNeymanAllocation",
    "KasySautmann",
    "RepresentationBandit",
]
