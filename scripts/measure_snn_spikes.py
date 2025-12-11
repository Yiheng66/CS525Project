import sys
import random
from typing import Tuple, Dict

import torch

# Prefer local copy of PyGame-Learning-Environment
sys.path.insert(0, "./PyGame-Learning-Environment")

from ple import PLE  # type: ignore
from ple.games.flappybird import FlappyBird  # type: ignore

from .eval_all_agents import (
    make_env,
    state_to_tensor,
    build_snn_agent,
    build_custom_snn_agent,
)


def measure_spikes_snn(
    name: str,
    policy_net,
    device: torch.device,
    num_samples: int = 200,
) -> Dict[str, float]:
    """
    Measure average firing rate alpha for a given SNN policy network.

    We collect `num_samples` states from random interaction with the environment
    and, for each state, perform a single forward pass through the SNN while
    requesting spike statistics.
    """
    env, action_dict, _ = make_env(display=False)

    total_spikes = 0
    total_neuron_slots = 0

    samples_collected = 0

    while samples_collected < num_samples:
        if env.game_over():
            env.reset_game()

        state_dict = env.getGameState()
        state = state_to_tensor(state_dict, device)
        state = state.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            # snnTorch SNN uses forward(x, return_spike_stats=True)
            q, stats = policy_net(state, return_spike_stats=True)

        total_spikes += stats["total_spikes"]
        total_neuron_slots += stats["total_neuron_slots"]
        samples_collected += 1

        # Take a random action to move the environment
        action_idx = random.randint(0, len(action_dict) - 1)
        env.act(action_dict[action_idx])

    alpha = total_spikes / float(total_neuron_slots) if total_neuron_slots > 0 else 0.0

    print(f"\n=== Spike statistics for {name} ===")
    print(f"Samples:           {num_samples}")
    print(f"Total spikes:      {total_spikes}")
    print(f"Total neuron slots:{total_neuron_slots}")
    print(f"Average firing rate alpha: {alpha:.4f}")

    return {
        "alpha": alpha,
        "total_spikes": total_spikes,
        "total_neuron_slots": total_neuron_slots,
    }


def measure_spikes_custom_snn(
    name: str,
    policy_net,
    device: torch.device,
    num_samples: int = 200,
) -> Dict[str, float]:
    """
    Measure average firing rate for the custom LIF SNN policy network.

    The custom SNN has forward(x, poisson_encode=False, return_spike_stats=True).
    """
    env, action_dict, _ = make_env(display=False)

    total_spikes = 0
    total_neuron_slots = 0
    samples_collected = 0

    while samples_collected < num_samples:
        if env.game_over():
            env.reset_game()

        state_dict = env.getGameState()
        state = state_to_tensor(state_dict, device)
        state = state.unsqueeze(0)

        with torch.no_grad():
            q, stats = policy_net(state, poisson_encode=False, return_spike_stats=True)

        total_spikes += stats["total_spikes"]
        total_neuron_slots += stats["total_neuron_slots"]
        samples_collected += 1

        action_idx = random.randint(0, len(action_dict) - 1)
        env.act(action_dict[action_idx])

    alpha = total_spikes / float(total_neuron_slots) if total_neuron_slots > 0 else 0.0

    print(f"\n=== Spike statistics for {name} ===")
    print(f"Samples:           {num_samples}")
    print(f"Total spikes:      {total_spikes}")
    print(f"Total neuron slots:{total_neuron_slots}")
    print(f"Average firing rate alpha: {alpha:.4f}")

    return {
        "alpha": alpha,
        "total_spikes": total_spikes,
        "total_neuron_slots": total_neuron_slots,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build snnTorch SNN agent and measure spikes
    base_env, action_dict, init_state = make_env(display=False)
    input_dim = len(init_state)
    n_actions = len(action_dict)
    # let base_env go out of scope

    snn_agent = build_snn_agent(input_dim, n_actions)
    measure_spikes_snn("SNN (snnTorch, T=25)", snn_agent.policy_net, snn_agent.device, num_samples=200)

    # Build custom LIF SNN agent and measure spikes
    custom_agent = build_custom_snn_agent(input_dim, n_actions)
    measure_spikes_custom_snn("Custom SNN (LIF, T=20)", custom_agent.policy_net, custom_agent.device, num_samples=200)


if __name__ == "__main__":
    main()

