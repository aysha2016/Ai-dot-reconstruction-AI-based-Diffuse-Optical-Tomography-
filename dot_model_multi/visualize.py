import matplotlib.pyplot as plt

def visualize_voxel_output(output_tensor):
    output_np = output_tensor.squeeze().detach().cpu().numpy()
    mid_slice = output_np[output_np.shape[0] // 2]
    plt.imshow(mid_slice, cmap='hot')
    plt.title("Mid-slice of Reconstructed Brain Activity")
    plt.colorbar()
    plt.show()