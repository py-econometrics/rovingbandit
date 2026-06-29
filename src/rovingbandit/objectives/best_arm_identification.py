"""Best-arm identification objective."""

from typing import Any

import numpy as np
from scipy import stats

from rovingbandit.core.exceptions import MissingConfigurationError
from rovingbandit.core.objective import Objective
from rovingbandit.core.policy import Policy
from rovingbandit.core.result import History


class BestArmIdentification(Objective):
    """
    Best-arm identification objective.

    Goal: Identify the best arm with high confidence using minimal samples.

    Reference: Even-Dar et al. (2006), Audibert & Bubeck (2010)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        n_mc_samples: int = 1000,
        seed: int | None = None,
    ) -> None:
        """
        Initialize best-arm identification objective.

        Args:
            confidence_threshold: Required confidence to stop (e.g., 0.95)
            n_mc_samples: Monte Carlo samples for computing confidence
            seed: Random seed for MC sampling
        """
        self.confidence_threshold = confidence_threshold
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(seed)

    def compute_metric(
        self,
        history: History,
        policy: Policy | None = None,
        true_best_arm: int | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute probability that identified arm is the best.

        Args:
            history: Complete history
            policy: Policy with current estimates
            true_best_arm: True best arm index (for evaluation)
            **kwargs: Ignored

        Returns:
            Confidence in best arm (probability it is best)
        """
        if policy is None:
            raise MissingConfigurationError("policy required for best-arm identification")

        confidence = self._compute_best_arm_probability(policy)
        return float(confidence)

    def stopping_criterion(self, policy: Policy, history: History, **kwargs: Any) -> bool:
        """
        Stop when confidence in best arm exceeds threshold.

        Args:
            policy: Current policy
            history: History so far
            **kwargs: Ignored

        Returns:
            True if confidence threshold met
        """
        confidence = self._compute_best_arm_probability(policy)
        return confidence >= self.confidence_threshold

    def get_metadata(self, policy: Policy, history: History, **kwargs: Any) -> dict[str, Any]:
        """Return confidence and best arm."""
        confidence = self._compute_best_arm_probability(policy)
        best_arm = int(np.argmax(policy.values))

        return {
            "confidence": float(confidence),
            "best_arm": best_arm,
        }

    def _compute_best_arm_probability(self, policy: Policy) -> float:
        """
        Compute probability that current best arm is truly best.

        Uses Monte Carlo sampling from posterior distributions.
        For Bernoulli rewards, assumes Beta posteriors.

        Args:
            policy: Policy with counts and values

        Returns:
            Probability that argmax(values) is the true best arm
        """
        n_arms = policy.n_arms
        counts = policy.counts
        values = policy.values

        # Check if this is a Thompson-style policy with success/failure counts
        successes = getattr(policy, "successes", None)
        failures = getattr(policy, "failures", None)
        if successes is not None and failures is not None:
            alpha = successes + 1
            beta = failures + 1
        else:
            # Approximate with Beta from empirical mean
            # Alpha = successes + 1, Beta = failures + 1
            # Mean = alpha / (alpha + beta) ≈ values
            # Use counts to back out alpha, beta
            alpha = values * counts + 1
            beta = (1 - values) * counts + 1
            # Clip to valid range
            alpha = np.maximum(alpha, 1.0)
            beta = np.maximum(beta, 1.0)

        # Monte Carlo sampling
        draws = np.zeros((self.n_mc_samples, n_arms))
        for arm in range(n_arms):
            draws[:, arm] = stats.beta.rvs(
                alpha[arm], beta[arm], size=self.n_mc_samples, random_state=self.rng
            )

        # Count how often each arm is best
        best_arm_samples = np.argmax(draws, axis=1)
        current_best = np.argmax(values)

        probability = np.mean(best_arm_samples == current_best)
        return float(probability)
