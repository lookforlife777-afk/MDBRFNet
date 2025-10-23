
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DualBranchUNet(nn.Module):
    def __init__(self, hsi_channels: int = 76, msi_channels: int = 3, depth_channels: int = 1,
                 num_classes: int = 5, feature_dim: int = 64):
        super(DualBranchUNet, self).__init__()

        self.feature_dim = feature_dim

        # HSI encoder
        self.hsi_enc1 = DoubleConv(hsi_channels, 64)
        self.hsi_enc2 = DoubleConv(64, 128)
        self.hsi_enc3 = DoubleConv(128, 256)
        self.hsi_enc4 = DoubleConv(256, 512)

        # MSI encoder
        self.msi_enc1 = DoubleConv(msi_channels, 64)
        self.msi_enc2 = DoubleConv(64, 128)
        self.msi_enc3 = DoubleConv(128, 256)
        self.msi_enc4 = DoubleConv(256, 512)

        # depth encoder (simplified)
        self.wd_dim = 32
        self.depth_enc = DoubleConv(depth_channels, self.wd_dim)

        # bottleneck and decoder
        self.bottleneck = DoubleConv(1024, 1024)
        self.dec4 = DoubleConv(1024 + 1024, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.feature_extractor = nn.Conv2d(64 + self.wd_dim, feature_dim, kernel_size=1)
        self.seg_head = nn.Conv2d(64 + self.wd_dim, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, hsi_x, msi_x, depth_x):
        hsi_e1 = self.hsi_enc1(hsi_x)
        hsi_e2 = self.hsi_enc2(self.pool(hsi_e1))
        hsi_e3 = self.hsi_enc3(self.pool(hsi_e2))
        hsi_e4 = self.hsi_enc4(self.pool(hsi_e3))

        msi_e1 = self.msi_enc1(msi_x)
        msi_e2 = self.msi_enc2(self.pool(msi_e1))
        msi_e3 = self.msi_enc3(self.pool(msi_e2))
        msi_e4 = self.msi_enc4(self.pool(msi_e3))

        combined_e4 = torch.cat([hsi_e4, msi_e4], dim=1)
        b = self.bottleneck(self.pool(combined_e4))

        d4 = self.dec4(torch.cat([self.upsample(b), combined_e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), hsi_e3 + msi_e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), hsi_e2 + msi_e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), hsi_e1 + msi_e1], dim=1))

        depth_e = self.depth_enc(depth_x)
        d1 = torch.cat([d1, depth_e], dim=1)

        features = self.feature_extractor(d1)
        seg_logits = self.seg_head(d1)
        return features, seg_logits
