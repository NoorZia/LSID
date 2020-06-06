from unet_model import UNet
from torch.nn import Module, PixelShuffle


class LSIDModel(Module):

    def __init__(self):
        super().__init__()
        self.convnet = UNet(in_channels=4, out_channels=12)
        self.upscaling = PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x1 = self.convnet(x)
        return self.upscaling(x1)
