import torch

data = torch.load('results/regime_samples/all_regimes_heston_params.pt', weights_only=False)

print("Available regimes in file:")
for regime in data['regimes'].keys():
    n_samples = len(data['regimes'][regime])
    print(f"  - {regime}: {n_samples} samples")
