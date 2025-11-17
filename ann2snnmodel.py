import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_DQN(nn.Module):
    """
    Spiking DQN model with surrogate gradients.
    Produces Q-values for DQN training.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128, 256, 512], beta=0.9, T=25):
        super().__init__()

        self.T = T
        self.beta = beta

        spike_grad = surrogate.fast_sigmoid()

        # ANN linear layers (same as your original DQN)
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            last_dim = h
        self.fc_layers = nn.ModuleList(layers)

        # SNN activation layers
        self.lif_layers = nn.ModuleList([
            snn.Leaky(beta=beta, spike_grad=spike_grad)
        for _ in hidden_dims])

        # Output layer (pure ANN)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)


    def init_states(self, batch_size, device):
        """Initialize membrane states for each LIF layer."""
        return [lif.init_hidden(batch_size, device=device) for lif in self.lif_layers]


    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Initialize membrane states
        mem_states = self.init_states(batch_size, device)

        # Accumulate Q-values across T timesteps
        q_accum = torch.zeros(batch_size, self.output_layer.out_features, device=device)

        for t in range(self.T):
            cur = x
            new_mem_states = []

            # Forward through each FC + LIF
            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_layers)):
                cur = fc(cur)
                spk, mem = lif(cur, mem_states[i])
                new_mem_states.append(mem)
                cur = spk  # spike output is next input

            mem_states = new_mem_states

            # Output Q-values based on last spiking representation
            q_accum += self.output_layer(cur)

        # Average Q-values over time
        return q_accum / self.T
