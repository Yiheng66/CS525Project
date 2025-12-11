import os
import sys
import time
import statistics
from typing import Tuple, List

import numpy as np
import torch

# Prefer local copy of PLE if present
sys.path.insert(0, "./PyGame-Learning-Environment")

from ple import PLE  # type: ignore
from ple.games.flappybird import FlappyBird  # type: ignore

from src import agent
from src import ssnagent
from src import ann2snnagent


def make_env(display: bool = False) -> Tuple[PLE, dict, dict]:
    """
    Create a FlappyBird environment and return:
      - PLE wrapper
      - action_dict: mapping from discrete index -> PLE action
      - init_state: initial game state dict
    """
    game = FlappyBird(width=256, height=256)
    env = PLE(game, display_screen=display)
    env.init()

    actions = env.getActionSet()
    # 0: flap, 1: do nothing (match main.py)
    action_dict = {0: actions[1], 1: actions[0]}
    state_dict = env.getGameState()
    return env, action_dict, state_dict


def state_to_tensor(state_dict: dict, device: torch.device) -> torch.Tensor:
    """Convert PLE game state dict to 1D float32 tensor on the given device."""
    return torch.tensor(list(state_dict.values()), dtype=torch.float32, device=device)


def build_ann_agent(input_dim: int, n_actions: int, weights_path: str, name: str) -> agent.Agent:
    """Create an ANN Dueling DQN-style agent for evaluation and load trained weights."""
    ann_agent = agent.Agent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=None,  # not used in evaluation loop
        EPS_START=0.0,
        EPS_END=0.0,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        network_type="DuelingDQN",
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{name} weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=ann_agent.device)
    ann_agent.policy_net.load_state_dict(state_dict)
    ann_agent.target_net.load_state_dict(state_dict)
    ann_agent.policy_net.eval()
    ann_agent.target_net.eval()
    return ann_agent


def build_snn_agent(input_dim: int, n_actions: int) -> ssnagent.SNNAgent:
    """Create an snnTorch-based SNN agent and load checkpoint if available."""
    snn_agent = ssnagent.SNNAgent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=None,  # not used in evaluation loop
        EPS_START=0.0,
        EPS_END=0.0,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        network_type="DuelingDQN",
        T=25,
    )

    # # Try to load full SNN checkpoint if present
    # snn_agent.load_checkpoint()
    # snn_agent.policy_net.eval()
    # snn_agent.target_net.eval()
    
    # Prefer custom SNN weights (trained via train_snn_snnTorch.py), fall back to ANN weights if needed.
    custom_weights_path = "models/snn_snntorch_policy.pt"
    ann_weights_path = "models/DuelingDQN_policy_net.pt"

    if os.path.exists(custom_weights_path):
        print(f"Custom SNN: loading SNN weights from {custom_weights_path}...")
        snn_state = torch.load(custom_weights_path, map_location=snn_agent.device)
        if not isinstance(snn_state, dict):
            snn_state = snn_state.state_dict()
        snn_agent.policy_net.load_state_dict(snn_state)
        snn_agent.target_net.load_state_dict(snn_state)
        print("Custom SNN: loaded trained SNN policy/target networks.")
    elif os.path.exists(ann_weights_path):
        print(f"Custom SNN: SNN weights not found; initialising from ANN weights at {ann_weights_path}...")
        ann_state = torch.load(ann_weights_path, map_location=snn_agent.device)
        if not isinstance(ann_state, dict):
            ann_state = ann_state.state_dict()

        snn_state = snn_agent.policy_net.state_dict()
        copied_count = 0
        for k, v in ann_state.items():
            if k in snn_state and snn_state[k].shape == v.shape:
                snn_state[k] = v
                copied_count += 1
        snn_agent.policy_net.load_state_dict(snn_state)
        snn_agent.target_net.load_state_dict(snn_state)
        print(f"Custom SNN: copied {copied_count} ANN layers into SNN policy/target networks.")
    else:
        print("Custom SNN: no pretrained weights found; evaluating from random initialisation.")

    snn_agent.policy_net.eval()
    snn_agent.target_net.eval()
    return snn_agent


def build_custom_snn_agent(input_dim: int, n_actions: int) -> ann2snnagent.SNNAgent:
    """Create a custom LIF SNN agent and initialise weights from custom SNN or ANN weights if available."""
    custom_agent = ann2snnagent.SNNAgent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=None,  # not used in evaluation loop
        EPS_START=0.0,
        EPS_END=0.0,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        T=20,
        network_type="DuelingDQN",
        device=ann2snnagent.device,
    )

    # Prefer custom SNN weights (trained via train_custom_snn_plot.py), fall back to ANN weights if needed.
    custom_weights_path = "models/snn_custom_policy.pt"
    ann_weights_path = "models/DuelingDQN_policy_net.pt"

    if os.path.exists(custom_weights_path):
        print(f"Custom SNN: loading SNN weights from {custom_weights_path}...")
        snn_state = torch.load(custom_weights_path, map_location=custom_agent.device)
        if not isinstance(snn_state, dict):
            snn_state = snn_state.state_dict()
        custom_agent.policy_net.load_state_dict(snn_state)
        custom_agent.target_net.load_state_dict(snn_state)
        print("Custom SNN: loaded trained SNN policy/target networks.")
    elif os.path.exists(ann_weights_path):
        print(f"Custom SNN: SNN weights not found; initialising from ANN weights at {ann_weights_path}...")
        ann_state = torch.load(ann_weights_path, map_location=custom_agent.device)
        if not isinstance(ann_state, dict):
            ann_state = ann_state.state_dict()

        snn_state = custom_agent.policy_net.state_dict()
        copied_count = 0
        for k, v in ann_state.items():
            if k in snn_state and snn_state[k].shape == v.shape:
                snn_state[k] = v
                copied_count += 1
        custom_agent.policy_net.load_state_dict(snn_state)
        custom_agent.target_net.load_state_dict(snn_state)
        print(f"Custom SNN: copied {copied_count} ANN layers into SNN policy/target networks.")
    else:
        print("Custom SNN: no pretrained weights found; evaluating from random initialisation.")

    custom_agent.policy_net.eval()
    custom_agent.target_net.eval()
    return custom_agent


def evaluate_agent(
    name: str,
    agent_obj,
    env: PLE,
    action_dict: dict,
    device: torch.device,
    episodes: int = 50,
    max_steps_per_episode: int = 2000,
) -> None:
    """Run a purely evaluative loop (no learning) and print summary stats."""
    scores: List[float] = []
    lengths: List[int] = []
    action_times: List[float] = []

    for ep in range(episodes):
        env.reset_game()
        state_dict = env.getGameState()
        state = state_to_tensor(state_dict, device)
        done = False
        steps = 0

        # Safety cap on episode length to avoid potential environment hangs
        while not done and steps < max_steps_per_episode:
            t0 = time.perf_counter()
            action_idx = agent_obj.take_action(state)
            t1 = time.perf_counter()
            action_times.append(t1 - t0)

            reward = env.act(action_dict[action_idx])
            done = env.game_over()
            if not done:
                next_state_dict = env.getGameState()
                state = state_to_tensor(next_state_dict, device)

            steps += 1

        scores.append(env.score())
        lengths.append(steps)

    mean_score = statistics.mean(scores)
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    mean_len = statistics.mean(lengths)
    std_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    mean_action_ms = 1000.0 * (statistics.mean(action_times) if action_times else 0.0)

    print(f"\n=== {name} ===")
    print(f"Episodes:            {episodes}")
    print(f"Average score:       {mean_score:.3f} ± {std_score:.3f}")
    print(f"Average steps/ep:    {mean_len:.2f} ± {std_len:.2f}")
    print(f"Mean action latency: {mean_action_ms:.3f} ms/decision")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Shared environment (we recreate between agents to avoid state carry-over)
    base_env, action_dict, init_state = make_env(display=False)
    input_dim = len(init_state)
    n_actions = len(action_dict)
    # PLE does not expose an explicit close() method; allow base_env to go out of scope.

    # Evaluate ANN baselines: Dueling DQN and (if available) Dueling DDQN
    env, action_dict, _ = make_env(display=False)
    ann_agent = build_ann_agent(input_dim, n_actions, "models/DuelingDQN_policy_net.pt", "Dueling DQN")
    evaluate_agent("ANN Dueling DQN", ann_agent, env, action_dict, ann_agent.device, episodes=50)

    if os.path.exists("models/DuelingDDQN_policy_net.pt"):
        ddqn_agent = build_ann_agent(input_dim, n_actions, "models/DuelingDDQN_policy_net.pt", "Dueling DDQN")
        evaluate_agent("ANN Dueling DDQN", ddqn_agent, env, action_dict, ddqn_agent.device, episodes=50)

    # Evaluate snnTorch-based SNN
    env, action_dict, _ = make_env(display=False)
    snn_agent = build_snn_agent(input_dim, n_actions)
    evaluate_agent("SNN (snnTorch, T=25)", snn_agent, env, action_dict, snn_agent.device, episodes=50)

    # Evaluate custom LIF SNN (ANN-to-SNN)
    env, action_dict, _ = make_env(display=False)
    custom_snn = build_custom_snn_agent(input_dim, n_actions)
    evaluate_agent("Custom SNN (LIF, T=20)", custom_snn, env, action_dict, custom_snn.device, episodes=50)


if __name__ == "__main__":
    main()
