import torch
import numpy as np
from typing import Tuple, Optional

class TissueProperties:
    def __init__(self):
        # Optical properties at different wavelengths (nm)
        self.wavelengths = torch.tensor([690, 780, 830])  # Common NIR wavelengths
        # Absorption coefficients (cm^-1) for different tissue types
        self.mua = {
            'background': torch.tensor([0.1, 0.15, 0.2]),  # Base absorption
            'activation': torch.tensor([0.3, 0.4, 0.5]),   # Activated region
            'vessel': torch.tensor([0.5, 0.6, 0.7])        # Blood vessel
        }
        # Scattering coefficients (cm^-1)
        self.mus = torch.tensor([10.0, 9.0, 8.0])  # Decreases with wavelength
        # Anisotropy factor
        self.g = 0.9

def create_phantom(voxel_dim: Tuple[int, int, int], 
                  tissue_props: TissueProperties,
                  activation_center: Optional[Tuple[int, int, int]] = None,
                  activation_radius: int = 5) -> torch.Tensor:
    """Create a 3D phantom with realistic tissue properties."""
    x, y, z = voxel_dim
    phantom = torch.zeros((len(tissue_props.wavelengths), x, y, z))
    
    # Add background absorption
    for wl_idx in range(len(tissue_props.wavelengths)):
        phantom[wl_idx] = tissue_props.mua['background'][wl_idx]
    
    # Add activation region if specified
    if activation_center is not None:
        cx, cy, cz = activation_center
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if (i-cx)**2 + (j-cy)**2 + (k-cz)**2 <= activation_radius**2:
                        for wl_idx in range(len(tissue_props.wavelengths)):
                            phantom[wl_idx, i, j, k] = tissue_props.mua['activation'][wl_idx]
    
    return phantom

def simulate_photon_migration(phantom: torch.Tensor, 
                            tissue_props: TissueProperties,
                            num_photons: int = 1000) -> torch.Tensor:
    """Simulate photon migration through tissue using a simplified Monte Carlo approach."""
    measurements = torch.zeros_like(phantom)
    x, y, z = phantom.shape[1:]
    
    # Simplified Monte Carlo simulation
    for wl_idx in range(len(tissue_props.wavelengths)):
        for _ in range(num_photons):
            # Start photon at random position on surface
            pos = torch.tensor([
                np.random.randint(0, x),
                np.random.randint(0, y),
                0  # Start at surface
            ])
            
            # Simulate photon path
            while 0 <= pos[2] < z:
                # Calculate step size based on total attenuation
                mua = phantom[wl_idx, pos[0], pos[1], pos[2]]
                mus = tissue_props.mus[wl_idx]
                mueff = torch.sqrt(3 * mua * (mua + mus * (1 - tissue_props.g)))
                step = -torch.log(torch.rand(1)) / mueff
                
                # Update position
                theta = 2 * np.pi * torch.rand(1)
                phi = torch.acos(2 * torch.rand(1) - 1)
                pos += step * torch.tensor([
                    torch.sin(phi) * torch.cos(theta),
                    torch.sin(phi) * torch.sin(theta),
                    torch.cos(phi)
                ])
                
                # Record measurement
                if 0 <= pos[0] < x and 0 <= pos[1] < y and 0 <= pos[2] < z:
                    measurements[wl_idx, int(pos[0]), int(pos[1]), int(pos[2])] += 1
    
    return measurements / num_photons

def simulate_mcx(voxel_dim: Tuple[int, int, int] = (32, 32, 32),
                num_wavelengths: int = 3,
                activation_center: Optional[Tuple[int, int, int]] = None,
                activation_radius: int = 5,
                num_photons: int = 1000) -> torch.Tensor:
    """
    Generate synthetic DOT measurements using a simplified Monte Carlo simulation.
    
    Args:
        voxel_dim: Dimensions of the 3D volume (x, y, z)
        num_wavelengths: Number of wavelengths to simulate
        activation_center: Center coordinates of activation region
        activation_radius: Radius of activation region
        num_photons: Number of photons to simulate per wavelength
    
    Returns:
        torch.Tensor: Simulated measurements of shape (num_wavelengths, *voxel_dim)
    """
    tissue_props = TissueProperties()
    phantom = create_phantom(voxel_dim, tissue_props, activation_center, activation_radius)
    measurements = simulate_photon_migration(phantom, tissue_props, num_photons)
    return measurements

if __name__ == "__main__":
    # Test the simulation
    measurements = simulate_mcx(
        voxel_dim=(32, 32, 32),
        activation_center=(16, 16, 16),
        activation_radius=5
    )
    print(f"Generated measurements shape: {measurements.shape}")
    print(f"Measurement range: [{measurements.min():.3f}, {measurements.max():.3f}]")