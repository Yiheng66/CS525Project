import sys
import time
import os

import torch
import matplotlib.pyplot as plt

# Prefer local copy of PyGame-Learning-Environment
sys.path.insert(0, "./PyGame-Learning-Environment")

from ple import PLE  # type: ignore
from ple.games.flappybird import FlappyBird  # type: ignore

import ann2snnagent


def make_env(display: bool = False) -> PLE:
    """Create a FlappyBird PLE environment."""
    game = FlappyBird(width=256, height=256)
    env = PLE(game, display_screen=display)
    env.init()
    return env


def get_state_vector(env: PLE) -> torch.Tensor:
    """Extract the 8-D feature vector from the PLE environment."""
    state_dict = env.getGameState()
    return torch.tensor(list(state_dict.values()), dtype=torch.float32)


def main() -> None:
    # -------------------------------
    # 1. Training configuration
    # -------------------------------
    NUM_EPISODES = 200  # reduced for faster experiment

    env = make_env(display=False)

    actions = env.getActionSet()
    action_dict = {0: actions[1], 1: actions[0]}
    n_actions = len(action_dict)

    test_input = get_state_vector(env)
    input_dim = test_input.shape[0]

    print(f"Custom SNN: state feature dimension = {input_dim}")

    # -------------------------------
    # 2. Create custom LIF SNN agent
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = ann2snnagent.SNNAgent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=action_dict,
        EPS_START=0.1,
        EPS_END=0.01,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        T=20,
        network_type="DuelingDQN",
        device=device,
    )

    # Load ANN weights into the custom SNN if available
    weights_path = "DuelingDDQN_policy_net.pt"
    if os.path.exists(weights_path):
        print(f"Custom SNN: loading ANN weights from {weights_path}...")
        ann_state = torch.load(weights_path, map_location=device)
        if not isinstance(ann_state, dict):
            ann_state = ann_state.state_dict()

        snn_state = agent.policy_net.state_dict()
        copied_count = 0
        for k, v in ann_state.items():
            if k in snn_state and snn_state[k].shape == v.shape:
                snn_state[k] = v
                copied_count += 1
        agent.policy_net.load_state_dict(snn_state)
        agent.target_net.load_state_dict(snn_state)
        print(f"Custom SNN: successfully copied {copied_count} layers from ANN.")
    else:
        print(f"Custom SNN: ANN weights not found at {weights_path}, starting from scratch.")

    # Disable per-episode plotting inside agent to speed up training
    def _noop_plot():
        return None

    agent.plot_durations = _noop_plot  # type: ignore[attr-defined]

    print(f"Training custom LIF SNN (T=20) for {NUM_EPISODES} episodes...")
    t0 = time.time()
    agent.steps_done = 0
    agent.episode_durations = []
    for ep in range(NUM_EPISODES):
        env.reset_game()
        state = get_state_vector(env).to(agent.device)
        done = False
        steps = 0
        while not done:
            action_idx = agent.take_action(state)
            reward = env.act(agent.action_dict[action_idx])
            reward_t = torch.tensor([reward], device=agent.device)
            action_t = torch.tensor([action_idx], device=agent.device)

            done = env.game_over()
            next_state = None if done else get_state_vector(env).to(agent.device)

            agent.cache_recall.cache(
                (state.cpu(), None if next_state is None else next_state.cpu(), action_t.cpu(), reward_t.cpu(), done)
            )

            state = next_state
            agent.optimize_model()
            agent.update_target_network()

            steps += 1

        agent.episode_durations.append(steps)

        if (ep + 1) % 1 == 0 or (ep + 1) == NUM_EPISODES:
            elapsed = time.time() - t0
            eps_per_min = (ep + 1) / (elapsed / 60.0)
            remaining = NUM_EPISODES - (ep + 1)
            eta_min = remaining / eps_per_min if eps_per_min > 0 else 0.0
            print(
                f"[Episode {ep+1}/{NUM_EPISODES}] "
                f"Last duration: {steps} steps | "
                f"Elapsed: {elapsed/60:.1f} min | ETA: {eta_min:.1f} min"
            )

    t1 = time.time()
    print(f"Custom SNN training finished in {(t1 - t0)/60:.2f} minutes.")

    # -------------------------------
    # 3. Plot custom SNN training curve
    # -------------------------------
    durations = torch.tensor(agent.episode_durations, dtype=torch.float)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(durations.numpy(), label="Episode duration", color="tab:blue", alpha=0.4)

    if len(durations) >= 100:
        means = durations.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(
            means.numpy(),
            label="100-episode moving average",
            color="tab:orange",
            linewidth=2.0,
        )

    plt.title("Custom LIF SNN Training on Flappy Bird (T=20)")
    plt.xlabel("Episode")
    plt.ylabel("Duration (steps)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("CustomSNN_training.png")
    print("Saved custom SNN training curve to CustomSNN_training.png")
    
    # Save model weights
    model_save_path = "CustomSNN_policy_net.pt"
    torch.save(agent.policy_net.state_dict(), model_save_path)
    print(f"Saved custom SNN model weights to {model_save_path}")


if __name__ == "__main__":
    main()
