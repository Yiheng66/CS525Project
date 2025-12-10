import sys
import time

import torch
import matplotlib.pyplot as plt

# Prefer local copy of PyGame-Learning-Environment
sys.path.insert(0, "./PyGame-Learning-Environment")

from ple import PLE  # type: ignore
from ple.games.flappybird import FlappyBird  # type: ignore
from itertools import count

import agent as ann_agent_mod


def make_env(display: bool = False) -> PLE:
    """Create a FlappyBird PLE environment."""
    game = FlappyBird(width=256, height=256)
    env = PLE(game, display_screen=display)
    env.init()
    return env


def train_with_progress(agent, env: PLE, num_episodes: int) -> None:
    """Custom training loop with simple progress logging."""
    agent.steps_done = 0
    start_time = time.time()

    for episode in range(num_episodes):
        env.reset_game()
        state_dict = env.getGameState()
        state = torch.tensor(list(state_dict.values()), dtype=torch.float32, device=agent.device)

        for c in count():
            action_idx = agent.take_action(state)
            reward = env.act(agent.action_dict[action_idx])
            reward_t = torch.tensor([reward], device=agent.device)
            action_t = torch.tensor([action_idx], device=agent.device)

            next_state_dict = env.getGameState()
            next_state = torch.tensor(list(next_state_dict.values()), dtype=torch.float32, device=agent.device)
            done = env.game_over()
            if done:
                next_state = None

            agent.cache_recall.cache((state, next_state, action_t, reward_t, done))
            state = next_state

            agent.optimize_model()
            agent.update_target_network()

            if done:
                agent.episode_durations.append(c + 1)
                break

        # Progress log every 100 episodes
        if (episode + 1) % 100 == 0 or (episode + 1) == num_episodes:
            elapsed = time.time() - start_time
            eps_done = episode + 1
            eps_per_min = eps_done / (elapsed / 60.0)
            remaining = num_episodes - eps_done
            eta_min = remaining / eps_per_min if eps_per_min > 0 else 0.0
            print(
                f"[Episode {eps_done}/{num_episodes}] "
                f"Last duration: {agent.episode_durations[-1]} steps | "
                f"Elapsed: {elapsed/60:.1f} min | ETA: {eta_min:.1f} min"
            )


def main() -> None:
    # -------------------------------
    # 1. Set training configuration
    # -------------------------------
    NUM_EPISODES = 10000  # You can lower to e.g. 2000 if training is slow

    env = make_env(display=False)
    state = env.getGameState()
    input_dim = len(state)

    actions = env.getActionSet()
    action_dict = {0: actions[1], 1: actions[0]}
    n_actions = len(action_dict)

    # -------------------------------
    # 2. Create ANN Dueling DDQN agent
    # -------------------------------
    agent = ann_agent_mod.Agent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim,
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=action_dict,
        EPS_START=1.0,
        EPS_END=0.05,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        network_type="DuelingDQN",
    )

    # Disable real-time plotting inside agent to speed up training;
    # we will generate a clean plot after training finishes.
    def _noop_plot():
        return None

    agent.plot_durations = _noop_plot  # type: ignore[attr-defined]

    print(f"Training Dueling DDQN for {NUM_EPISODES} episodes...")
    t0 = time.time()
    train_with_progress(agent, env, NUM_EPISODES)
    t1 = time.time()
    print(f"Training finished in {(t1 - t0)/60:.2f} minutes.")

    # -------------------------------
    # 3. Create a high-quality training curve
    # -------------------------------
    durations = torch.tensor(agent.episode_durations, dtype=torch.float)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(durations.numpy(), label="Episode duration", color="tab:blue", alpha=0.4)

    if len(durations) >= 100:
        means = durations.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="100-episode moving average", color="tab:orange", linewidth=2.0)

    plt.title("Dueling DDQN Training on Flappy Bird")
    plt.xlabel("Episode")
    plt.ylabel("Duration (steps)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("DuelingDDQN_training_10000_episodes.png")
    print("Saved training curve to DuelingDDQN_training.png")


if __name__ == "__main__":
    main()
