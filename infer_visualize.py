import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D
from dataset import BrainMRIDataset
from torch.utils.data import DataLoader

def inference_and_visualize():
    # Load the trained model
    model = UNet3D(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('unet3d_model.pth'))
    model.eval()

    # Load a test sample
    test_dataset = BrainMRIDataset(root_dir='path/to/test/data')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Get a single test sample
    test_sample = next(iter(test_loader))
    input_modalities, target = test_sample

    # Perform inference
    with torch.no_grad():
        output = model(input_modalities)

    # Convert tensors to numpy arrays
    input_np = input_modalities.squeeze().numpy()
    target_np = target.squeeze().numpy()
    output_np = output.squeeze().numpy()

    # Visualize results
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    slice_idx = input_np.shape[1] // 2  # Middle slice

    # Input modalities
    for i in range(3):
        axes[0, i].imshow(input_np[i, slice_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Input Modality {i+1}')
        axes[0, i].axis('off')

    # Target
    axes[1, 1].imshow(target_np[slice_idx, :, :], cmap='gray')
    axes[1, 1].set_title('Target')
    axes[1, 1].axis('off')

    # Output
    axes[2, 1].imshow(output_np[slice_idx, :, :], cmap='gray')
    axes[2, 1].set_title('Model Output')
    axes[2, 1].axis('off')

    # Difference (Target - Output)
    diff = target_np - output_np
    axes[3, 1].imshow(diff[slice_idx, :, :], cmap='bwr', vmin=-1, vmax=1)
    axes[3, 1].set_title('Difference (Target - Output)')
    axes[3, 1].axis('off')

    # Remove empty subplots
    for i in [0, 2]:
        fig.delaxes(axes[1, i])
        fig.delaxes(axes[2, i])
        fig.delaxes(axes[3, i])

    plt.tight_layout()
    plt.savefig('inference_results.png')
    plt.show()

if __name__ == '__main__':
    inference_and_visualize()