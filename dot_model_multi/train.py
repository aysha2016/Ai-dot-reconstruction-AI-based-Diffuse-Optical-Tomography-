import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from simulate_mcx import simulate_mcx, TissueProperties
from visualize import visualize_slice_comparison

class DOTDataset(Dataset):
    def __init__(self, num_samples: int, voxel_dim: Tuple[int, int, int] = (32, 32, 32)):
        self.num_samples = num_samples
        self.voxel_dim = voxel_dim
        self.tissue_props = TissueProperties()
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random activation center
        activation_center = (
            np.random.randint(5, self.voxel_dim[0]-5),
            np.random.randint(5, self.voxel_dim[1]-5),
            np.random.randint(5, self.voxel_dim[2]-5)
        )
        activation_radius = np.random.randint(3, 8)
        
        # Generate measurements and ground truth
        measurements = simulate_mcx(
            voxel_dim=self.voxel_dim,
            activation_center=activation_center,
            activation_radius=activation_radius,
            num_photons=1000
        )
        
        # Ground truth is the activation map (first wavelength)
        ground_truth = measurements[0].unsqueeze(0)
        
        return measurements, ground_truth

def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                num_epochs: int = 10,
                batch_size: int = 4,
                learning_rate: float = 1e-3,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                save_dir: Optional[str] = None) -> dict:
    """
    Train the 3D U-Net model for DOT reconstruction.
    
    Args:
        model: The 3D U-Net model
        optimizer: Optimizer for training
        criterion: Loss function
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints and logs
    
    Returns:
        dict: Training history
    """
    # Create save directory if specified
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = save_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save training configuration
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': device,
            'model_architecture': str(model)
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    # Move model to device
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = DOTDataset(num_samples=1000)
    val_dataset = DOTDataset(num_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf')
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if save_dir is not None and val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, run_dir / 'best_model.pth')
            
            # Visualize some validation results
            if epoch % 5 == 0:
                sample_inputs, sample_targets = next(iter(val_loader))
                sample_outputs = model(sample_inputs.to(device))
                visualize_slice_comparison(
                    sample_targets[0],
                    sample_outputs[0].cpu(),
                    slice_idx=sample_targets.shape[2] // 2
                )
                plt.savefig(run_dir / f'validation_epoch_{epoch}.png')
                plt.close()
    
    # Save final model and history
    if save_dir is not None:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, run_dir / 'final_model.pth')
        
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=4)
    
    return history

if __name__ == "__main__":
    from unet3d import UNet3D
    
    # Test training
    model = UNet3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=2,  # Short test run
        batch_size=2,
        save_dir='runs'
    )
    
    print("Training completed!")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")