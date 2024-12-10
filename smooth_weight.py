from smooth.permute import PermutationManager, compute_tv_loss_for_network, PermutationManager_SingleTSP, PermutationManager_SingleTSP_Greedy, PermutationManager_Random
import torch
from pathlib import Path
import argparse
from neumeta.models.lenet import MnistNet, MnistResNet

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth pretrained weights")
    parser.add_argument('--model', type=str, required=True, help='Path to the pretrained model')

    args = parser.parse_args()
    return args

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create the model for CIFAR10
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
pretrained_weight = args.model
hidden_dim = 32
model = MnistNet(hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(pretrained_weight))
model.eval()  # Set to evaluation mode

# Compute the total variation loss for the network
total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
print("Total Total Variation After Training:", total_tv)

# Apply permutations to the model's layers and check the total variation
input_tensor = torch.randn(1, 1, 28, 28).to(device)
permute_func = PermutationManager(model, input_tensor)
# Compute the permutation matrix for each clique graph, save as a dict
permute_dict = permute_func.compute_permute_dict()
# Apply permutation to the weight
model = permute_func.apply_permutations(permute_dict, ignored_keys=[])
total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
print("Total Total Variation After Permute:", total_tv)
pretrained_weight = pretrained_weight.split(".")[0]
torch.save(model.state_dict(), f"{pretrained_weight}_permute.pth")
