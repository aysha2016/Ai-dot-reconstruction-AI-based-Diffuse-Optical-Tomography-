import torch
import torch.nn as nn
from simulate_mcx import simulate_mcx
from unet3d import UNet3D

def train_model(model, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        input_data = simulate_mcx()
        labels = torch.rand_like(input_data[:1])  # Synthetic ground truth
        optimizer.zero_grad()
        output = model(input_data.unsqueeze(0))
        loss = criterion(output, labels.unsqueeze(0))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")