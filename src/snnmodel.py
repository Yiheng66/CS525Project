import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNN_DQN(nn.Module):
    def __init__(self, ann_model, beta=0.9, T=25):
        super().__init__()
        self.T = T
        spike_grad = surrogate.fast_sigmoid()

        # Copy ANN layers directly
        self.layer1 = ann_model.layer1
        self.layer2 = ann_model.layer2
        self.layer3 = ann_model.layer3
        self.layer4 = ann_model.layer4
        self.layer5 = ann_model.layer5
        
        if ann_model.network_type == 'DuelingDQN':
            self.state_values = ann_model.state_values
            self.advantages = ann_model.advantages
        else:
            self.output = ann_model.output

        # Add spiking neurons
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, return_spike_stats: bool = False):
        batch = x.size(0)

        mem1 = torch.zeros(batch, self.layer1.out_features, device=x.device)
        mem2 = torch.zeros(batch, self.layer2.out_features, device=x.device)
        mem3 = torch.zeros(batch, self.layer3.out_features, device=x.device)
        mem4 = torch.zeros(batch, self.layer4.out_features, device=x.device)
        mem5 = torch.zeros(batch, self.layer5.out_features, device=x.device)

        out_sum = 0
        total_spikes = 0

        for _ in range(self.T):
            spk1, mem1 = self.lif1(self.layer1(x), mem1)
            spk2, mem2 = self.lif2(self.layer2(spk1), mem2)
            spk3, mem3 = self.lif3(self.layer3(spk2), mem3)
            spk4, mem4 = self.lif4(self.layer4(spk3), mem4)
            spk5, mem5 = self.lif5(self.layer5(spk4), mem5)

            if return_spike_stats:
                total_spikes += (
                    spk1.sum() + spk2.sum() + spk3.sum() + spk4.sum() + spk5.sum()
                )

            if hasattr(self, "state_values"):
                v = self.state_values(spk5)
                a = self.advantages(spk5)
                q = v + (a - torch.max(a, dim=1, keepdim=True)[0])
            else:
                q = self.output(spk5)

            out_sum += q

        q_avg = out_sum / self.T

        if not return_spike_stats:
            return q_avg

        # Total number of neuron slots across all hidden layers and timesteps
        hidden_neurons = (
            self.layer1.out_features
            + self.layer2.out_features
            + self.layer3.out_features
            + self.layer4.out_features
            + self.layer5.out_features
        )
        total_neuron_slots = batch * self.T * hidden_neurons
        return q_avg, {
            "total_spikes": total_spikes.item(),
            "total_neuron_slots": int(total_neuron_slots),
            "T": self.T,
        }
