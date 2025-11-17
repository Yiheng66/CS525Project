# ann2snnmain.py
"""
Main for training the SNNAgent on FlappyBird using Feature inputs (Option A).
This matches the architecture of the provided DuelingDQN_policy_net.pt (8 inputs).
"""

import sys
import os
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import torch

# local imports
import ann2snnagent

def get_state_vector(env):
    """
    Extracts the game state features (8 values) instead of an image.
    """
    state_dict = env.getGameState()
    return torch.tensor(list(state_dict.values()), dtype=torch.float32)

def main():
    game = FlappyBird(width=256, height=256)
    # display_screen=True helps you see if it's working, set to False for speed
    p = PLE(game, display_screen=True, force_fps=30) 
    p.init()

    # action set (two actions: no-op, flap)
    actions = p.getActionSet()
    action_dict = {0: actions[1], 1: actions[0]}
    n_actions = len(action_dict)

    # Get input dimension from the state vector (Should be 8)
    test_input = get_state_vector(p)
    input_dim = test_input.shape[0]
    
    print(f"State Vector Size: {input_dim} (Expecting 8)")

    # Initialize Agent
    agent = ann2snnagent.SNNAgent(
        BATCH_SIZE=32,
        MEMORY_SIZE=100000,
        GAMMA=0.99,
        input_dim=input_dim, # This will now be 8, matching your .pt file
        output_dim=n_actions,
        action_dim=n_actions,
        action_dict=action_dict,
        EPS_START=0.1,       # Start with low exploration since we are loading pre-trained weights
        EPS_END=0.01,
        EPS_DECAY_VALUE=0.999995,
        lr=1e-4,
        TAU=0.005,
        T=20,
        network_type='DuelingDQN',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ------------------------------------------------------
    # Load ANN weights
    # ------------------------------------------------------
    weights_path = "DuelingDQN_policy_net.pt"
    
    if os.path.exists(weights_path):
        print(f"Loading ANN weights from {weights_path}...")
        ann_state = torch.load(weights_path, map_location=agent.device)
        
        # If the .pt file is the full model, extract state_dict
        if not isinstance(ann_state, dict):
            ann_state = ann_state.state_dict()

        snn_state = agent.policy_net.state_dict()

        copied_count = 0
        for k, v in ann_state.items():
            if k in snn_state:
                # Ensure shapes match before copying
                if snn_state[k].shape == v.shape:
                    snn_state[k] = v
                    copied_count += 1
                else:
                    print(f"Skipping {k}: Shape mismatch. ANN={v.shape}, SNN={snn_state[k].shape}")
        
        agent.policy_net.load_state_dict(snn_state)
        # Sync target net with the newly loaded weights
        agent.target_net.load_state_dict(snn_state) 
        
        print(f"Successfully loaded {copied_count} layers from ANN to SNN.")
    else:
        print(f"Warning: {weights_path} not found. Starting from scratch.")


    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------
    N_EPISODES = 1000000
    
    # We pass the get_state_vector function as the preprocessor
    agent.train(
        episodes=N_EPISODES, 
        env=p, 
        render=True, 
        preprocess_fn=get_state_vector
    )

if __name__ == "__main__":
    main()