import torch
from unet3d import UNet3D
from simulate_mcx import simulate_mcx
from train import train_model
from visualize import visualize_voxel_output

import torch.nn as nn

if __name__ == "__main__":
    model = UNet3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_model(model, optimizer, criterion, num_epochs=5)
    output = model(simulate_mcx().unsqueeze(0))
    visualize_voxel_output(output)