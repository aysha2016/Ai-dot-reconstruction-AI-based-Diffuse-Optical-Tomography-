import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet3D, self).__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.bottleneck = ConvBlock(64, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        self.out_conv = nn.Conv3d(32, out_channels, 1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.out_conv(dec1)