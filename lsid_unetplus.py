import torch
import torch.nn as nn
from torch.nn import Module, PixelShuffle



class conv_block_nested(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        output = self.activation(x)

        return output

class Expand(nn.Module):
    """ Upsampling with transpose convolution, concatenate, then two consecutive convolutions. """

    def __init__(self, in_channels):
        super().__init__()
        self.expand = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
       

    def forward(self, x_current):
        x_current = self.expand(x_current)        
        return x_current


class UNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

     
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up1 = Expand(128)
        self.Up2 = Expand(256)
        self.Up3 = Expand(128)
        self.Up4 = Expand(512)
        self.Up5 = Expand(256)
        self.Up6 = Expand(128)
        self.Up7 = Expand(1024)
        self.Up8 = Expand(512)
        self.Up9 = Expand(256)
        self.Up10 = Expand(128)
        
        self.conv0_0 = conv_block_nested(in_ch, 64)
        self.conv1_0 = conv_block_nested(64, 128)
        self.conv2_0 = conv_block_nested(128, 256)
        self.conv3_0 = conv_block_nested(256, 512)
        self.conv4_0 = conv_block_nested(512, 1024)

        self.conv0_1 = conv_block_nested(64 + 128, 64)
        self.conv1_1 = conv_block_nested(128 + 256, 128)
        self.conv2_1 = conv_block_nested(256 + 512, 256)
        self.conv3_1 = conv_block_nested(512 + 1024, 512)

        self.conv0_2 = conv_block_nested(64*2 + 128, 64)
        self.conv1_2 = conv_block_nested(128*2 + 256, 128)
        self.conv2_2 = conv_block_nested(256*2 + 512, 256)

        self.conv0_3 = conv_block_nested(64*3 + 128, 64)
        self.conv1_3 = conv_block_nested(128*3 + 256, 128)

        self.conv0_4 = conv_block_nested(64*4 + 128, 64)

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up2(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up3(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up4(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up5(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up6(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up7(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up8(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up9(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up10(x1_3)], 1))

        output = self.final(x0_4)
        return output

        
class LSIDplusModelconv(Module):

    def __init__(self):
        super().__init__()
        self.convnet = UNet(in_ch=4, out_ch=12)
        self.upscaling = PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x1 = self.convnet(x)
        return self.upscaling(x1)
