import sys, os
sys.path.append("./PyGame-Learning-Environment")
import pygame
from ple import PLE
from ple.games.flappybird import FlappyBird
import ssnagent

game = FlappyBird(width=256, height=256)
env = PLE(game, display_screen=False)
env.init()
action_dict = {0: env.getActionSet()[1], 1: env.getActionSet()[0]}
state = env.getGameState()

input_dim = len(state)
output_dim = len(action_dict)

agent = ssnagent.SNNAgent(
    BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99,
    input_dim=input_dim, output_dim=output_dim, action_dim=output_dim,
    action_dict=action_dict, EPS_START=1.0, EPS_END=0.05, 
    EPS_DECAY_VALUE=0.999995, TAU=0.005, lr=1e-4, 
    network_type='DuelingDQN', T=25
)

# Try to load checkpoint and resume training
start_episode = agent.load_checkpoint()

if start_episode > 0:
    print(f"\nResuming training from episode {start_episode}...")

agent.train(episodes=10000000, env=env, start_episode=start_episode)