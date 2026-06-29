"""Objective functions for bandit algorithms."""

from abc import ABC, abstractmethod
from typing import Any

from rovingbandit.core.policy import Policy
from rovingbandit.core.result import History


class Objective(ABC):
    """
    Abstract base class for bandit objectives.

    Defines what the bandit algorithm is trying to achieve.
    """

    @abstractmethod
    def compute_metric(self, history: History, **kwargs: Any) -> float:
        """
        Compute objective-specific performance metric.

        Args:
            history: Complete history of the run
            **kwargs: Additional objective-specific parameters

        Returns:
            Performance metric value
        """
        pass

    @abstractmethod
    def stopping_criterion(self, policy: Policy, history: History, **kwargs: Any) -> bool:
        """
        Determine if objective is satisfied (for early stopping).

        Args:
            policy: Current policy state
            history: Complete history so far
            **kwargs: Additional objective-specific parameters

        Returns:
            True if stopping criterion met, False otherwise
        """
        pass

    def get_metadata(self, policy: Policy, history: History, **kwargs: Any) -> dict[str, Any]:
        """
        Get objective-specific metadata for results.

        Args:
            policy: Final policy state
            history: Complete history
            **kwargs: Additional parameters

        Returns:
            Dictionary of metadata
        """
        return {}
