import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from dataset import BrainMRIDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import UNet3D
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import gc
import psutil

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def main():
    root_dir = '../../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cpu")
    print("Using CPU for training")

    model = UNet3D(in_channels=3, out_channels=1).to(device)
    model.use_checkpointing = True  # Enable gradient checkpointing

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (modalities, _) in enumerate(tqdm(train_loader)):
            inputs = modalities[:, :3, :, :, :].to(device)  # T1ce, T2, FLAIR
            target = modalities[:, 3, :, :, :].to(device)  # T1

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}")
                print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Target shape: {target.shape}")
                print_memory_usage()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (modalities, _) in enumerate(tqdm(test_loader)):
                inputs = modalities[:, :3, :, :, :].to(device)
                target = modalities[:, 3, :, :, :].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, target.unsqueeze(1))
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(test_loader)}")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()  # This won't affect CPU memory, but keeping it for completeness
        print_memory_usage()

    # Save the model
    torch.save(model.state_dict(), 'unet3d_model.pth')


if __name__ == '__main__':
    main()