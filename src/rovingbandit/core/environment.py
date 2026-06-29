"""Bandit environment implementation."""

from collections.abc import Callable

import numpy as np

from rovingbandit.core.exceptions import (
    InvalidArmError,
    InvalidConfigurationError,
    MissingConfigurationError,
)


class BanditEnvironment:
    """
    Represents a multi-armed bandit environment.

    Attributes:
        n_arms: Number of available arms
        arm_means: True means for each arm (if known/simulated)
        costs: Cost per pull for each arm
        arm_groups: Group membership for representation constraints
        reward_fn: Custom reward function (arm_idx, rng) -> reward
    """

    def __init__(
        self,
        n_arms: int,
        arm_means: np.ndarray | None = None,
        costs: np.ndarray | None = None,
        arm_groups: np.ndarray | None = None,
        reward_fn: Callable[[int, np.random.Generator], float] | None = None,
        contexts: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize bandit environment.

        Args:
            n_arms: Number of arms
            arm_means: Mean reward for each arm (for Bernoulli bandits)
            costs: Cost to pull each arm (default: all 1.0)
            arm_groups: Group membership array for representation constraints
            reward_fn: Custom reward function, overrides arm_means if provided
            contexts: Optional matrix of contextual features per arm (n_arms x d)
            seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.arm_means = arm_means
        self.costs = costs if costs is not None else np.ones(n_arms)
        self.arm_groups = arm_groups
        self.reward_fn = reward_fn
        self.contexts = contexts
        self.rng = np.random.default_rng(seed)

        # Validate inputs
        if arm_means is not None:
            if len(arm_means) != n_arms:
                raise InvalidConfigurationError(
                    f"arm_means has length {len(arm_means)}, expected n_arms={n_arms}"
                )
            if not np.all((arm_means >= 0) & (arm_means <= 1)):
                raise InvalidConfigurationError("arm_means must lie in [0, 1]")

        if costs is not None:
            if len(costs) != n_arms:
                raise InvalidConfigurationError(
                    f"costs has length {len(costs)}, expected n_arms={n_arms}"
                )
            if not np.all(costs > 0):
                raise InvalidConfigurationError("costs must be strictly positive")

        if arm_groups is not None and len(arm_groups) != n_arms:
            raise InvalidConfigurationError(
                f"arm_groups has length {len(arm_groups)}, expected n_arms={n_arms}"
            )

        if contexts is not None and contexts.shape[0] != n_arms:
            raise InvalidConfigurationError(
                f"contexts first dimension is {contexts.shape[0]}, expected n_arms={n_arms}"
            )

    def pull(self, arm: int) -> tuple[float, float]:
        """
        Pull an arm and observe reward.

        Args:
            arm: Index of arm to pull (0 to n_arms-1)

        Returns:
            Tuple of (reward, cost)
        """
        if not 0 <= arm < self.n_arms:
            raise InvalidArmError(f"arm {arm} out of range [0, {self.n_arms})")

        if self.reward_fn is not None:
            reward = self.reward_fn(arm, self.rng)
        elif self.arm_means is not None:
            # Bernoulli reward
            reward = float(self.rng.binomial(1, self.arm_means[arm]))
        else:
            raise MissingConfigurationError("either arm_means or reward_fn must be provided")

        cost = float(self.costs[arm])
        return reward, cost

    def get_optimal_arm(self, objective: str = "reward") -> int:
        """
        Get the optimal arm for a given objective.

        Args:
            objective: Either 'reward', 'cost', or 'reward_per_cost'

        Returns:
            Index of optimal arm
        """
        if self.arm_means is None:
            raise MissingConfigurationError("cannot determine optimal arm without arm_means")

        if objective == "reward":
            return int(np.argmax(self.arm_means))
        elif objective == "cost":
            return int(np.argmin(self.costs))
        elif objective == "reward_per_cost":
            return int(np.argmax(self.arm_means / self.costs))
        else:
            raise InvalidConfigurationError(f"unknown objective: {objective!r}")

    def get_optimal_reward(self, objective: str = "reward") -> float:
        """Get the expected reward from optimal arm."""
        if self.arm_means is None:
            raise MissingConfigurationError("cannot determine optimal reward without arm_means")

        optimal_arm = self.get_optimal_arm(objective)
        return float(self.arm_means[optimal_arm])

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the random number generator."""
        self.rng = np.random.default_rng(seed)

    def get_contexts(self) -> np.ndarray | None:
        """Return contextual feature matrix if available."""
        return None if self.contexts is None else self.contexts.copy()
