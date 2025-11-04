import torch
from model import DQN

# --- Configuration ---
# adjust based on how you trained your model
network_type = 'DuelingDQN'     # or 'DQN' / 'DDQN'
input_dim = 8                   # number of input features (len(state))
output_dim = 2                  # number of possible actions
weights_path = f"{network_type}_policy_net.pt"  # or target_net if you prefer

# --- Load model ---
model = DQN(input_dim=input_dim, output_dim=output_dim, network_type=network_type)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# --- Print all weights layer by layer ---
print(f"\nLoaded weights from: {weights_path}\n")
for name, param in model.named_parameters():
    print(f"{name} — shape: {tuple(param.shape)}")
    print(param.data)
    print("-" * 80)
