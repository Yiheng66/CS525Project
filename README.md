# Comparing Spiking Neural Networks and Artificial Neural Networks for Real-Time Game Control: A Study on Flappy Bird

This repository contains implementations of ANN-based and SNN-based deep reinforcement learning agents trained to play Flappy Bird. The project compares three approaches: a Dueling Double DQN baseline (ANN), an snnTorch-based SNN, and a custom LIF SNN with surrogate gradients.

## Overview

**Key Results:**
- The ANN Dueling DDQN achieves near-perfect performance (score 57.0) with minimal latency (~0.25 ms)
- The Custom LIF SNN matches ANN performance when initialized with pre-trained weights (~4 ms latency)
- Theoretical energy analysis suggests SNNs could reduce per-decision energy by 60-75% on neuromorphic hardware

## Installation

### Requirements
- Python 3.7+
- See `requirements.txt` for dependencies


## Project Structure

```
├── src/                           # Core implementation
│   ├── model.py                   # ANN Dueling DQN architecture
│   ├── agent.py                   # ANN training agent
│   ├── snnmodel.py                # snnTorch-based SNN
│   ├── ssnagent.py                # snnTorch SNN training agent
│   ├── ann2snnmodel.py            # Custom LIF SNN architecture
│   ├── ann2snnagent.py            # Custom LIF SNN training agent
│   └── ann2snnmain.py             # ANN-to-SNN weight transfer
│
├── scripts/                       # User-facing execution scripts
│   ├── train_ddqn_plot.py         # Train ANN Dueling DDQN
│   ├── train_snn_snnTorch_plot.py # Train snnTorch-based SNN
│   ├── train_custom_snn_plot.py   # Train custom LIF SNN
│   ├── eval_agents.py             # Evaluate all agents
│   ├── measure_spike_activity.py  # Calculate spike statistics for energy analysis
│   └── eval_snn_T_sweep.py        # Evaluate SNNs across different timestep horizons
│
├── models/                        # Pre-trained model checkpoints
│   ├── DuelingDDQN_policy_net.pt  # Trained ANN policy network
│   ├── DuelingDDQN_target_net.pt  # Trained ANN target network
│   ├── CustomSNN_policy_net.pt    # Trained custom LIF SNN policy network
│   └── SNN_snnTorch_policy_net.pt # Trained snnTorch SNN policy network
│
├── plots/                         # Generated training curves and results
│   ├── DuelingDDQN_training.png
│   ├── SNN_snnTorch_training.png
│   ├── CustomSNN_training.png
│   └── Flappy_Bird_edit.gif
│
├── PyGame-Learning-Environment/   # Flappy Bird game environment
└── requirements.txt               # Python dependencies
```

## Quick Start

### Option 1: Evaluate Pre-trained Models

To evaluate all three pre-trained agents:

```bash
python -m scripts.eval_agents
```

This will:
- Load the pre-trained ANN, snnTorch SNN, and custom LIF SNN models
- Run 50 evaluation episodes for each agent
- Print performance metrics (average score, steps, latency)
- Save results to console

### Option 2: Train from Scratch

#### Train the ANN Baseline
```bash
python -m scripts.train_ddqn_plot
```
Trains a Dueling Double DQN agent for 2000 episodes. Saves:
- Trained policy and target networks to `models/DuelingDDQN_*.pt`
- Training curve to `plots/DuelingDDQN_training.png`

#### Train the snnTorch-based SNN
```bash
python -m scripts.train_snn_snnTorch_plot
```
Trains an snnTorch SNN initialized from the pre-trained Dueling DDQN. Saves:
- Trained networks to `models/SNN_snnTorch_*.pt`
- Training curve to `plots/SNN_snnTorch_training.png`

#### Train the Custom LIF SNN
```bash
python -m scripts.train_custom_snn_plot
```
Trains a custom LIF SNN with surrogate gradients, initialized from the pre-trained Dueling DDQN. Saves:
- Trained networks to `models/CustomSNN_*.pt`
- Training curve to `plots/CustomSNN_training.png`

### Option 3: Measure Spike Activity (For Energy Analysis)

```bash
python -m scripts.measure_spike_activity
```

This script:
- Loads trained SNN models
- Measures average firing rates (α) across hidden layers using random sampled states
- Outputs spike statistics used for theoretical energy cost calculations
- Results used in equations (21)–(24) from the paper

### Option 4: Evaluate SNNs Across Timestep Horizons

```bash
python -m scripts.eval_snn_T_sweep
```

Evaluates the custom LIF SNN with different T values to analyze latency vs. performance trade-offs.

## Configuration & Hyperparameters

Currently, hyperparameters are hard-coded in the training scripts. To modify training behavior:

1. **ANN Training** (`train_ddqn_plot.py`): Edit hyperparameters like learning rate, batch size, epsilon decay
2. **SNN Training** (`train_snn_snnTorch_plot.py`, `train_custom_snn_plot.py`): Edit timesteps (T), leak factor (β), threshold (V_th)
3. **Agent Evaluation** (`eval_agents.py`): Adjust number of evaluation episodes and episode length cap

See Tables I and II in the paper for default hyperparameter values.

## Expected Output

### From `eval_agents.py`:
```
========== ANN (DDQN) Evaluation ==========
Average Score: 57.0
Average Steps: 2000
Inference Latency: 0.25 ms

========== SNN (snnTorch) Evaluation ==========
Average Score: 1.36
Average Steps: 252.18
Inference Latency: 17.08 ms

========== Custom SNN (LIF) Evaluation ==========
Average Score: 57.0
Average Steps: 2000
Inference Latency: 4.39 ms
```

### From `measure_spike_activity.py`:
```
Average firing rate (α) for snnTorch SNN: 0.10
Average firing rate (α) for Custom LIF SNN: 0.03

Theoretical cost (Custom SNN): 0.65 × ANN_cost
```

## Key Findings

| Agent | Performance (Score) | Latency (ms) | Theoretical Cost |
|-------|---------------------|--------------|------------------|
| ANN Dueling DDQN | 57.0 | 0.25 | 1.0× |
| SNN snnTorch (T=25) | 1.36 | 17.08 | 2.5× |
| Custom LIF SNN (T=20) | 57.0 | 4.39 | 0.65× |

**Key Insights:**
- Custom LIF SNNs match ANN performance via weight transfer without sacrificing accuracy
- On standard CPUs, SNNs are slower due to timestep unrolling overhead
- On neuromorphic hardware, SNNs could achieve 60–75% energy reduction
- For real-time 60 Hz control, all agents remain viable (<16.7 ms latency)