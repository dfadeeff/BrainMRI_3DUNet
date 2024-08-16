import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


def center_crop(layer, target_size):
    _, _, layer_depth, layer_height, layer_width = layer.size()
    diff_z = (layer_depth - target_size[0]) // 2
    diff_y = (layer_height - target_size[1]) // 2
    diff_x = (layer_width - target_size[2]) // 2
    return layer[:, :, diff_z:diff_z + target_size[0], diff_y:diff_y + target_size[1], diff_x:diff_x + target_size[2]]


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)

        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]
        print(f"Input shape: {x.shape}")
        enc1 = self.encoder1(x)
        print(f"enc1 shape: {enc1.shape}")
        enc2 = self.encoder2(self.pool1(enc1))
        print(f"enc2 shape: {enc2.shape}")
        enc3 = self.encoder3(self.pool2(enc2))
        print(f"enc3 shape: {enc3.shape}")
        bottleneck = self.bottleneck(self.pool3(enc3))
        print(f"bottleneck shape: {bottleneck.shape}")

        dec3 = self.upconv3(bottleneck)
        print(f"dec3 shape before concat: {dec3.shape}")
        dec3 = self.decoder3(torch.cat([dec3, center_crop(enc3, dec3.shape[2:])], dim=1))
        print(f"dec3 shape after concat: {dec3.shape}")
        dec2 = self.upconv2(dec3)
        print(f"dec2 shape before concat: {dec2.shape}")
        dec2 = self.decoder2(torch.cat([dec2, center_crop(enc2, dec2.shape[2:])], dim=1))
        print(f"dec2 shape after concat: {dec2.shape}")
        dec1 = self.upconv1(dec2)
        print(f"dec1 shape before concat: {dec1.shape}")
        dec1 = self.decoder1(torch.cat([dec1, center_crop(enc1, dec1.shape[2:])], dim=1))
        print(f"dec1 shape after concat: {dec1.shape}")

        output = self.conv(dec1)
        print(f"Output shape before interpolation: {output.shape}")

        # Interpolate the output to match the input size
        output = F.interpolate(output, size=input_size, mode='trilinear', align_corners=False)
        print(f"Final output shape: {output.shape}")
        return output