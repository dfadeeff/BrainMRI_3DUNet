import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D
from dataset import BrainMRIDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import torch.nn as nn


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def inference_and_visualize():
    output_dir = 'inference_results'
    ensure_dir(output_dir)

    root_dir = '../../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    full_dataset = BrainMRIDataset(root_dir=root_dir)

    _, test_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet3D(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('unet3d_model.pth'))
    model.eval()

    criterion = nn.MSELoss()

    for i, test_sample in enumerate(tqdm(test_loader, desc="Processing patients")):
        patient_id = test_indices[i]

        input_modalities, _ = test_sample

        print(f"\nProcessing patient ID: {patient_id}")
        print(f"Original input shape: {input_modalities.shape}")

        # Select only the first 3 modalities (T1ce, T2, FLAIR) as input
        input_modalities = input_modalities[:, :3, :, :, :]
        target = test_sample[0][:, 3:4, :, :, :]  # T1 as target

        print(f"Input shape: {input_modalities.shape}")
        print(f"Target shape: {target.shape}")

        # Perform inference
        with torch.no_grad():
            output = model(input_modalities)

        print(f"Output shape: {output.shape}")

        # Calculate loss
        loss = criterion(output, target).item()
        print(f"Loss: {loss}")

        # Convert tensors to numpy arrays
        input_np = input_modalities.squeeze().numpy()
        target_np = target.squeeze().numpy()
        output_np = output.squeeze().numpy()

        print(f"Input numpy shape: {input_np.shape}")
        print(f"Target numpy shape: {target_np.shape}")
        print(f"Output numpy shape: {output_np.shape}")

        # Visualize results
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        slice_idx = input_np.shape[1] // 2  # Middle slice

        # Input modalities
        modality_names = ['T1ce', 'T2', 'FLAIR']
        for j in range(3):
            axes[0, j].imshow(input_np[j, slice_idx, :, :], cmap='gray')
            axes[0, j].set_title(f'Input: {modality_names[j]}')
            axes[0, j].axis('off')

        # Target (T1)
        if target_np.ndim == 3:
            axes[1, 1].imshow(target_np[slice_idx, :, :], cmap='gray')
        elif target_np.ndim == 4:
            axes[1, 1].imshow(target_np[0, slice_idx, :, :], cmap='gray')
        else:
            print(f"Unexpected target shape: {target_np.shape}")
            axes[1, 1].text(0.5, 0.5, 'Unexpected target shape', ha='center', va='center')
        axes[1, 1].set_title('Target: T1')
        axes[1, 1].axis('off')

        # Output
        if output_np.ndim == 3:
            axes[2, 1].imshow(output_np[slice_idx, :, :], cmap='gray')
        elif output_np.ndim == 4:
            axes[2, 1].imshow(output_np[0, slice_idx, :, :], cmap='gray')
        else:
            print(f"Unexpected output shape: {output_np.shape}")
            axes[2, 1].text(0.5, 0.5, 'Unexpected output shape', ha='center', va='center')
        axes[2, 1].set_title('Model Output')
        axes[2, 1].axis('off')

        # Difference (Target - Output)
        if target_np.shape == output_np.shape:
            diff = target_np - output_np
            if diff.ndim == 3:
                axes[3, 1].imshow(diff[slice_idx, :, :], cmap='bwr', vmin=-1, vmax=1)
            elif diff.ndim == 4:
                axes[3, 1].imshow(diff[0, slice_idx, :, :], cmap='bwr', vmin=-1, vmax=1)
        else:
            print(f"Cannot compute difference. Shapes: target {target_np.shape}, output {output_np.shape}")
            axes[3, 1].text(0.5, 0.5, 'Cannot compute difference', ha='center', va='center')
        axes[3, 1].set_title('Difference (Target - Output)')
        axes[3, 1].axis('off')

        # Remove empty subplots
        for j in [0, 2]:
            fig.delaxes(axes[1, j])
            fig.delaxes(axes[2, j])
            fig.delaxes(axes[3, j])

        plt.tight_layout()
        plt.suptitle(f'Patient ID: {patient_id}, Loss: {loss:.4f}')
        plt.savefig(os.path.join(output_dir, f'inference_results_patient_{patient_id}.png'))
        plt.close()


if __name__ == '__main__':
    inference_and_visualize()