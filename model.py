import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling block for U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling block for U-Net."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use a simple upsampling followed by a conv, else use transpose conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Note: in_channels is the concatenated channels from skip connection + upsampled
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 if needed (ensure dimensions match)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final convolution layer for U-Net."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MiniUNet(nn.Module):
    """A lightweight U-Net model for segmentation."""
    def __init__(self, n_channels=1, n_classes=1, base_filters=16, bilinear=True, depth=3):
        super(MiniUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        
        # Initial double convolution
        self.inc = DoubleConv(n_channels, base_filters)
        
        # Downsampling path
        self.down_path = nn.ModuleList()
        
        # Store feature map sizes for skip connections
        self.feature_channels = [base_filters]
        
        # Create downsampling blocks
        in_channels = base_filters
        for i in range(depth):
            out_channels = in_channels * 2
            self.down_path.append(Down(in_channels, out_channels))
            self.feature_channels.append(out_channels)
            in_channels = out_channels
        
        # Upsampling path
        self.up_path = nn.ModuleList()
        
        # Create upsampling blocks
        for i in range(depth):
            # Get input channels for upsampling block
            in_features = self.feature_channels[-(i+1)]
            out_features = self.feature_channels[-(i+2)]
            
            # Total channels after concatenation
            total_channels = in_features + out_features
            
            self.up_path.append(Up(total_channels, out_features, bilinear))
        
        # Final convolution
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        # Initial feature extraction
        x1 = self.inc(x)
        
        # Downsampling path - save skip connections
        skip_connections = [x1]
        x_down = x1
        
        for down in self.down_path:
            x_down = down(x_down)
            skip_connections.append(x_down)
        
        # Remove the last feature map from skip connections - it's the bottleneck
        x_up = skip_connections.pop()
        
        # Upsampling path - use skip connections
        for up in self.up_path:
            x2 = skip_connections.pop()
            x_up = up(x_up, x2)
        
        # Final convolution
        logits = self.outc(x_up)
        
        return logits

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Return Dice loss
        return 1 - dice