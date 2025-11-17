# snnagent.py
"""
Spiking Neural Network Agent (DQN style) using a simple custom LIF + surrogate gradient.
Designed to be drop-in for your FlappyBird project that previously used agent.py.
Uses your MemoryRecall.py for replay memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from itertools import count
import pygame as pg

# import your MemoryRecall module (same as agent.py)
import MemoryRecall

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Surrogate spike function
# -----------------------------
class SurrogateSpike(torch.autograd.Function):
    """
    Forward: binary spike (x >= 0 -> 1), else 0
    Backward: fast sigmoid surrogate derivative
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (input >= 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # fast sigmoid surrogate derivative
        gamma = 0.3
        grad = grad_output * gamma / (1.0 + input.abs())**2
        return grad

spike_fn = SurrogateSpike.apply

# -----------------------------
# Simple LIF layer (per-neuron)
# -----------------------------
class LIFLayer(nn.Module):
    def __init__(self, size, beta=0.9, threshold=1.0, device='cpu'):
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold
        self.device = device

    def init_state(self, batch_size):
        # membrane potential (batch, neurons)
        return torch.zeros(batch_size, self.size, device=self.device)

    def forward_step(self, current, mem):
        """
        current: (batch, neurons) input current (pre-activation)
        mem: previous membrane potential (batch, neurons)
        returns: spike (batch,neurons), new_mem (batch,neurons)
        """
        # integrate
        new_mem = self.beta * mem + current
        # spike based on threshold
        mem_minus_thr = new_mem - self.threshold
        spike = spike_fn(mem_minus_thr)
        # reset membrane on spike (simple reset to 0)
        new_mem = new_mem * (1.0 - spike)
        return spike, new_mem

# -----------------------------
# SNN DQN model (mirrors your ANN DQN)
# -----------------------------
class SNN_DQN(nn.Module):
    def __init__(self, input_dim, output_dim, beta=0.9, T=20, device='cpu', network_type='DuelingDQN'):
        """
        input_dim: dimensionality of flattened input
        output_dim: number of actions
        T: number of SNN timesteps per decision
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.T = T

        # Linear layers (same sizes as your ANN)
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512, 512)

        # LIF layers corresponding to each dense layer's output size
        self.lif1 = LIFLayer(64, beta=beta, device=self.device)
        self.lif2 = LIFLayer(128, beta=beta, device=self.device)
        self.lif3 = LIFLayer(256, beta=beta, device=self.device)
        self.lif4 = LIFLayer(512, beta=beta, device=self.device)
        self.lif5 = LIFLayer(512, beta=beta, device=self.device)

        if network_type == 'DuelingDQN':
            self.state_values = nn.Linear(512, 1)
            self.advantages = nn.Linear(512, output_dim)
        else:
            self.output = nn.Linear(512, output_dim)

        self.to(self.device)

    def forward(self, x, poisson_encode=False):
        """
        x: (batch, input_dim)
        poisson_encode: 
            If True, inputs must be 0..1 (converts to spike probabilities).
            If False, inputs are treated as constant current (Direct Encoding).
        """
        batch = x.shape[0]

        # initialize membrane potentials
        mem1 = self.lif1.init_state(batch)
        mem2 = self.lif2.init_state(batch)
        mem3 = self.lif3.init_state(batch)
        mem4 = self.lif4.init_state(batch)
        mem5 = self.lif5.init_state(batch)

        q_accum = torch.zeros(batch, self.output_dim, device=self.device)

        for t in range(self.T):
            # Encoding: either Poisson spike coding or repeated analog input
            if poisson_encode:
                # Only use this for normalized images (0 to 1)
                inp_spikes = torch.bernoulli(x).to(self.device)
                cur1 = self.layer1(inp_spikes)
            else:
                # Direct Encoding (Constant Current) for Features (values > 1)
                cur1 = self.layer1(x)

            spk1, mem1 = self.lif1.forward_step(cur1, mem1)
            cur2 = self.layer2(spk1)
            spk2, mem2 = self.lif2.forward_step(cur2, mem2)
            cur3 = self.layer3(spk2)
            spk3, mem3 = self.lif3.forward_step(cur3, mem3)
            cur4 = self.layer4(spk3)
            spk4, mem4 = self.lif4.forward_step(cur4, mem4)
            cur5 = self.layer5(spk4)
            spk5, mem5 = self.lif5.forward_step(cur5, mem5)

            if self.network_type == 'DuelingDQN':
                state_val = self.state_values(spk5)
                adv = self.advantages(spk5)
                q_t = state_val + (adv - adv.max(1, keepdim=True)[0])
            else:
                q_t = self.output(spk5)

            q_accum += q_t

        q_avg = q_accum / float(self.T)
        return q_avg


# -----------------------------
# Agent (DQN-style) using SNN_DQN
# -----------------------------
class SNNAgent:
    def __init__(self,
                 BATCH_SIZE,
                 MEMORY_SIZE,
                 GAMMA,
                 input_dim,
                 output_dim,
                 action_dim,
                 action_dict,
                 EPS_START,
                 EPS_END,
                 EPS_DECAY_VALUE,
                 lr,
                 TAU,
                 T=20,
                 network_type='DuelingDQN',
                 device=device):
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.GAMMA = GAMMA
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY_VALUE = EPS_DECAY_VALUE
        self.eps = EPS_START
        self.TAU = TAU
        self.device = device
        self.episode_durations = []

        self.cache_recall = MemoryRecall.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type = network_type

        # create policy and target SNNs
        self.policy_net = SNN_DQN(input_dim=input_dim, output_dim=output_dim, beta=0.9, T=T, network_type=network_type, device=device).to(self.device)
        self.target_net = SNN_DQN(input_dim=input_dim, output_dim=output_dim, beta=0.9, T=T, network_type=network_type, device=device).to(self.device)
        # freeze target params gradients (target net not directly optimized)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # copy params
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0

    @torch.no_grad()
    def take_action(self, state):
        # state is torch tensor (input_dim,)
        self.eps = self.eps * self.EPS_DECAY_VALUE
        self.eps = max(self.eps, self.EPS_END)

        if self.eps < np.random.rand():
            s = state.unsqueeze(0).to(self.device)
            # CRITICAL FIX: Set poisson_encode=False because we are using Features (>1.0)
            qvals = self.policy_net(s, poisson_encode=False)
            action_idx = torch.argmax(qvals, dim=1).item()
        else:
            action_idx = random.randint(0, self.action_dim - 1)
        self.steps_done += 1
        return action_idx

    def update_target_network(self):
        # soft update target params: target = tau*policy + (1-tau)*target
        target_dict = self.target_net.state_dict()
        policy_dict = self.policy_net.state_dict()
        for k in policy_dict.keys():
            target_dict[k] = policy_dict[k] * self.TAU + target_dict[k] * (1.0 - self.TAU)
        self.target_net.load_state_dict(target_dict)

    def optimize_model(self):
        if len(self.cache_recall) < self.BATCH_SIZE:
            return
        batch = self.cache_recall.recall(self.BATCH_SIZE)
        batch = [*zip(*batch)]
        state_batch = torch.stack(batch[0]).to(self.device)            # (B, input_dim)
        next_states = batch[1]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(self.device) if any(non_final_mask) else torch.empty(0, device=self.device)
        action_batch = torch.stack(batch[2]).to(self.device)           # (B,1)
        reward_batch = torch.cat(batch[3]).to(self.device)            # (B,)

        # predicted Q for the actions taken
        # CRITICAL FIX: Set poisson_encode=False here too
        state_action_values = self.policy_net(state_batch, poisson_encode=False).gather(1, action_batch)

        # compute next state values from target
        next_state_values = torch.zeros(self.BATCH_SIZE, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, poisson_encode=False).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping (optional but helpful for stability)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def plot_durations(self):
        # same plotting as agent.py, minimal here (optional)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('SNN Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)
        plt.savefig(self.network_type + '_SNN_training.png')

    def train(self, episodes, env, render=False, preprocess_fn=None):
        """
        episodes: number of episodes to train
        env: PLE environment (p)
        preprocess_fn: function that converts env screen -> torch.tensor([input_dim])
        """
        self.steps_done = 0

        for episode in range(episodes):
            env.reset_game()
            # initial observation
            if preprocess_fn is None:
                raise ValueError("preprocess_fn required for SNN agent")
            state = preprocess_fn(env)
            state = state.to(self.device)
            for c in count():
                action_idx = self.take_action(state)
                reward = env.act(self.action_dict[action_idx])
                reward = torch.tensor([reward], device=self.device)
                action = torch.tensor([action_idx], device=self.device)
                done = env.game_over()
                next_state = None if done else preprocess_fn(env).to(self.device)
                # store transition (state, next_state, action, reward, done)
                self.cache_recall.cache((state.cpu(), None if next_state is None else next_state.cpu(), action.cpu(), reward.cpu(), done))
                state = next_state
                self.optimize_model()
                self.update_target_network()

                if render:
                    pg.display.update()

                if done:
                    self.episode_durations.append(c + 1)
                    print(f"Episode {episode} | Steps: {c+1} | Score: {env.score()} | Eps: {self.eps:.3f}")
                    # save networks
                    torch.save(self.target_net.state_dict(), self.network_type + '_SNN_target_net.pt')
                    torch.save(self.policy_net.state_dict(), self.network_type + '_SNN_policy_net.pt')
                    self.plot_durations()
                    break