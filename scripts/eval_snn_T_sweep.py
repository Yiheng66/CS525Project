import os
import sys
import time
import statistics

import torch

# Prefer local copy of PyGame-Learning-Environment
sys.path.insert(0, "./PyGame-Learning-Environment")

from ple import PLE  # type: ignore
from ple.games.flappybird import FlappyBird  # type: ignore

from src import ssnagent


def make_env(display: bool = False) -> PLE:
    game = FlappyBird(width=256, height=256)
    env = PLE(game, display_screen=display)
    env.init()
    return env


def state_to_tensor(state_dict: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(state_dict.values()), dtype=torch.float32, device=device)


def evaluate_snn(T: int, episodes: int = 20, max_steps: int = 2000) -> None:
    env = make_env(display=False)
    actions = env.getActionSet()
    action_dict = {0: actions[1], 1: actions[0]}
    n_actions = len(action_dict)

    state_dict = env.getGameState()
    input_dim = len(state_dict)

    agent = ssnagent.SNNAgent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=action_dict,
        EPS_START=0.0,  # eval only
        EPS_END=0.0,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        network_type="DuelingDQN",
        T=T,
    )

    # Load the same checkpoint used in main experiments, if available
    agent.load_checkpoint()
    agent.policy_net.eval()

    device = agent.device

    scores = []
    lengths = []
    latencies = []

    for ep in range(episodes):
        env.reset_game()
        s_dict = env.getGameState()
        state = state_to_tensor(s_dict, device)
        done = False
        steps = 0

        while not done and steps < max_steps:
            t0 = time.perf_counter()
            action_idx = agent.take_action(state)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            env.act(action_dict[action_idx])
            done = env.game_over()
            if not done:
                s_dict = env.getGameState()
                state = state_to_tensor(s_dict, device)
            steps += 1

        scores.append(env.score())
        lengths.append(steps)

    mean_score = statistics.mean(scores)
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    mean_steps = statistics.mean(lengths)
    std_steps = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    mean_latency_ms = 1000.0 * (statistics.mean(latencies) if latencies else 0.0)

    print(f"\n=== snnTorch SNN T={T} ===")
    print(f"Episodes:            {episodes}")
    print(f"Average score:       {mean_score:.3f} ± {std_score:.3f}")
    print(f"Average steps/ep:    {mean_steps:.2f} ± {std_steps:.2f}")
    print(f"Mean action latency: {mean_latency_ms:.3f} ms/decision")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try a few different T values; adjust as needed
    T_values = [10, 20, 25, 30]
    for T in T_values:
        evaluate_snn(T=T, episodes=20, max_steps=2000)


if __name__ == "__main__":
    main()

