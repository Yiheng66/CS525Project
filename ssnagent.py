import torch
import snnmodel
import MemoryRecall
import random
import numpy as np
import model
import torch.optim as optim
from itertools import count
import matplotlib.pyplot as plt 
import pygame as pg
import math

class SNNAgent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU, network_type='DDQN', T=25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY_VALUE = EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        
        self.network_type = network_type # <-- FIX: Save the network_type attribute
        self.episode_durations = [] # <-- FIX: Add list for plotting

        # replay buffer
        self.memory = MemoryRecall.MemoryRecall(memory_size=MEMORY_SIZE)

        # Create ANN base + wrap into SNN
        
        ann_policy = model.DQN(input_dim, output_dim, network_type)
        ann_target = model.DQN(input_dim, output_dim, network_type)

        self.policy_net = snnmodel.SNN_DQN(ann_model=ann_policy, beta=0.9, T=T).to(self.device)
        self.target_net = snnmodel.SNN_DQN(ann_model=ann_target, beta=0.9, T=T).to(self.device)

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    def take_action(self, state):
        # epsilon decay
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY_VALUE)

        if np.random.rand() > self.eps:
            state = state[None, :]
            state = state.to(self.device) # <-- FIX: Move state tensor to the correct device
            with torch.no_grad():
                action_idx = torch.argmax(self.policy_net(state), dim=1).item()
        else:
            action_idx = random.randint(0, self.action_dim - 1)

        return action_idx

    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        #Plot the durations
        plt.plot(durations_t.numpy())
        # Take 100 episode averages of the durations and plot them too, to show a running average on the graph
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        # Use the SNN-specific name for saving the plot
        plt.savefig(self.network_type + '_SNN_training.png')

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = self.memory.recall(self.BATCH_SIZE)
        batch = [*zip(*batch)]

        state = torch.stack(batch[0])
        non_final_mask = torch.tensor(tuple(s is not None for s in batch[1]), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch[1] if s is not None])
        action = torch.stack(batch[2])
        reward = torch.cat(batch[3])

        # Q(s, a)
        state_action_values = self.policy_net(state).gather(1, action)

        # max_a' Q_target(s', a')
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward

        loss = torch.nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for param_tgt, param_src in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_tgt.data.copy_(param_tgt.data * (1 - self.TAU) + param_src.data * self.TAU)

    def train(self, episodes, env):
        for episode in range(episodes):
            env.reset_game()
            state = env.getGameState()
            state = torch.tensor(list(state.values()), dtype=torch.float32, device=self.device)

            for t in count():
                action = self.take_action(state)
                reward = env.act(self.action_dict[action])
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action], device=self.device)

                next_state = env.getGameState()
                if next_state is not None:
                    next_state = torch.tensor(list(next_state.values()), dtype=torch.float32, device=self.device)
                
                done = env.game_over()

                if done:
                    next_state = None

                self.memory.cache((state, next_state, action, reward, done))
                state = next_state

                self.optimize_model()
                self.update_target_network()
                pg.display.update()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    
                    print(f"Episode {episode} | Steps: {t+1} | Score: {env.score()} | Eps: {self.eps:.3f}")
                    
                    network_type = self.network_type + '_SNN' 
                    torch.save(self.target_net.state_dict(), network_type + '_target_net.pt')
                    torch.save(self.policy_net.state_dict(), network_type + '_policy_net.pt')
                    break