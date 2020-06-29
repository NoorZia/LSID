import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-net implementation based on https://github.com/milesial/Pytorch-UNet
  
    """

    def __init__(self, in_channels=4, out_channels=12):
        super().__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels

        # Two 3x3 convolutions
        self.incoming = DoubleConv2d(in_channels=in_channels, out_channels=64)

        # Each Module: Two 3x3 convolutions + 2x2 Max pooling
        self.contract1 = Contract(in_channels=64, out_channels=128)
        self.contract2 = Contract(in_channels=128, out_channels=256)
        self.contract3 = Contract(in_channels=256, out_channels=512)
        self.contract4 = Contract(in_channels=512, out_channels=1024)

        # Each Module: Transpose 2x2 convolution + two 3x3 convolutions
        self.expand1 = Expand(in_channels=1024, out_channels=512)
        self.expand2 = Expand(in_channels=512, out_channels=256)
        self.expand3 = Expand(in_channels=256, out_channels=128)
        self.expand4 = Expand(in_channels=128, out_channels=64)

        # 1x1 convolution
        self.outgoing = Outgoing(in_channels=64, out_channels=out_channels)

    def forward(self, x):
        # Input layer
        x1 = self.incoming(x)

        # Contractive part
        x2 = self.contract1(x1)
        x3 = self.contract2(x2)
        x4 = self.contract3(x3)
        x5 = self.contract4(x4)

        # Expansive part, with skip connections
        x = self.expand1(x_current=x5, x_old=x4)
        x = self.expand2(x, x_old=x3)
        x = self.expand3(x, x_old=x2)
        x = self.expand4(x, x_old=x1)

        # Output layer
        output = self.outgoing(x)

        return output


# U-net modules below
class DoubleConv2d(nn.Module):
    """ Two consecutive convolutions with ReLU activations. """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv2d = nn.Sequential(
            # NOTE: Original U-net uses padding=0, but we don't need tiling.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch-norm in some implementations?
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv2d(x)


class Contract(nn.Module):
    """ Downsampling with MaxPooling, then two consecutive convolutions """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.contract = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.contract(x)


class Expand(nn.Module):
    """ Upsampling with transpose convolution, concatenate, then two consecutive convolutions. """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv2d = DoubleConv2d(in_channels, out_channels)

    def forward(self, x_current, x_old):
        x_current = self.expand(x_current)
        x_concat = torch.cat([x_old, x_current], dim=1)  # dim=1: concatenate along Channel-dimension
        return self.double_conv2d(x_concat)


class Outgoing(nn.Module):
    """ 1x1 convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
