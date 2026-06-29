"""Cross-cutting end-to-end workflows spanning policies, objectives, and runners."""

import numpy as np

from rovingbandit import (
    UCB1,
    BanditEnvironment,
    BestArmIdentification,
    GreedyPolicy,
    OnlineRunner,
    RandomPolicy,
    RegretMinimization,
    ThompsonSampling,
)


def test_regret_minimization_workflow():
    env = BanditEnvironment(
        n_arms=5,
        arm_means=np.array([0.1, 0.3, 0.5, 0.4, 0.2]),
        seed=42,
    )
    policies = {
        "Random": RandomPolicy(n_arms=5, seed=42),
        "Greedy": GreedyPolicy(n_arms=5, seed=42),
        "UCB1": UCB1(n_arms=5, seed=42),
        "Thompson": ThompsonSampling(n_arms=5, seed=42),
    }
    objective = RegretMinimization(optimal_reward=0.5)
    runner = OnlineRunner()

    results = {}
    for name, policy in policies.items():
        env.reset_rng(42)
        results[name] = runner.run(policy, env, n_steps=500, objective=objective)

    for result in results.values():
        assert result.final_regret is not None
        assert result.final_regret >= 0

    # A good adaptive policy beats the random baseline on regret.
    assert results["Thompson"].final_regret < results["Random"].final_regret


def test_best_arm_identification_workflow():
    env = BanditEnvironment(
        n_arms=3,
        arm_means=np.array([0.3, 0.5, 0.9]),
        seed=42,
    )
    policy = ThompsonSampling(n_arms=3, seed=42)
    objective = BestArmIdentification(confidence_threshold=0.9, n_mc_samples=500, seed=42)
    runner = OnlineRunner()

    result = runner.run(policy, env, n_steps=1000, objective=objective, early_stopping=True)

    assert result.n_steps <= 1000
    assert result.metadata["confidence"] >= 0.9
    assert result.metadata["best_arm"] == 2
