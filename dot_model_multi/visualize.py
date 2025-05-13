import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import torch

def visualize_voxel_output(output_tensor: torch.Tensor,
                         alpha: float = 0.1,
                         threshold: float = 0.1,
                         colormap: str = 'hot') -> None:
    """
    Render 3D voxel data using matplotlib volume rendering with interactive controls.
    
    Args:
        output_tensor: 3D tensor of shape (D, H, W) or (1, D, H, W)
        alpha: Opacity of the volume rendering
        threshold: Minimum value to display
        colormap: Colormap to use for visualization
    """
    # Convert to numpy and remove batch dimension if present
    volume = output_tensor.squeeze().detach().cpu().numpy()
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create slider for threshold control
    ax_threshold = plt.axes([0.2, 0.02, 0.6, 0.03])
    threshold_slider = Slider(ax_threshold, 'Threshold', 0, 1, valinit=threshold)
    
    # Create slider for alpha control
    ax_alpha = plt.axes([0.2, 0.06, 0.6, 0.03])
    alpha_slider = Slider(ax_alpha, 'Opacity', 0, 1, valinit=alpha)
    
    def update(val):
        ax.clear()
        current_threshold = threshold_slider.val
        current_alpha = alpha_slider.val
        
        # Create voxel grid
        x, y, z = np.indices(volume.shape)
        mask = volume > current_threshold
        
        # Plot voxels
        ax.voxels(x, y, z, mask, facecolors=plt.cm.get_cmap(colormap)(volume),
                 alpha=current_alpha, edgecolor='none')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Blood Oxygenation Map')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        plt.draw()
    
    # Register update function with sliders
    threshold_slider.on_changed(update)
    alpha_slider.on_changed(update)
    
    # Initial plot
    update(None)
    
    # Add colorbar
    norm = plt.Normalize(volume.min(), volume.max())
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Oxygenation Level')
    
    plt.show()

def visualize_slice_comparison(ground_truth: torch.Tensor,
                             prediction: torch.Tensor,
                             slice_idx: int = None) -> None:
    """
    Visualize a comparison between ground truth and prediction slices.
    
    Args:
        ground_truth: Ground truth tensor of shape (D, H, W) or (1, D, H, W)
        prediction: Prediction tensor of shape (D, H, W) or (1, D, H, W)
        slice_idx: Index of the slice to visualize (if None, shows middle slice)
    """
    # Convert to numpy and remove batch dimension if present
    gt = ground_truth.squeeze().detach().cpu().numpy()
    pred = prediction.squeeze().detach().cpu().numpy()
    
    if slice_idx is None:
        slice_idx = gt.shape[0] // 2
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot ground truth
    im1 = ax1.imshow(gt[slice_idx], cmap='hot')
    ax1.set_title('Ground Truth')
    plt.colorbar(im1, ax=ax1)
    
    # Plot prediction
    im2 = ax2.imshow(pred[slice_idx], cmap='hot')
    ax2.set_title('Prediction')
    plt.colorbar(im2, ax=ax2)
    
    # Plot difference
    diff = np.abs(gt[slice_idx] - pred[slice_idx])
    im3 = ax3.imshow(diff, cmap='coolwarm')
    ax3.set_title('Absolute Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(f'Slice {slice_idx} Comparison')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test visualization with random data
    test_volume = torch.rand(32, 32, 32)
    visualize_voxel_output(test_volume)
    
    # Test slice comparison
    test_gt = torch.rand(32, 32, 32)
    test_pred = test_gt + 0.1 * torch.randn_like(test_gt)
    visualize_slice_comparison(test_gt, test_pred)