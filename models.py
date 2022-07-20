import torch
from torch import nn
import torch.nn.functional as F
from involution import *

class RDB(nn.Module):
    """
    Achieves densely connected convolutional layers, called Residual Dense Block
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(RDB, self).__init__()
        self.channels = channels
        self.growths = growths

        self.conv_list = nn.ModuleList([self.make_conv(i) for i in range(4)])
        self.conv4 = get_conv(channels + growths * 4, channels)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()



    def make_conv(self, i):
        conv = nn.Conv2d(self.channels + self.growths * i, self.growths, (3, 3), (1, 1), (1, 1))
        return conv

    def forward(self, x):
        identity = x

        out1 = self.lrelu(self.conv_list[0](x))
        in2 = torch.cat([x, out1], 1)
        out2 = self.lrelu(self.conv_list[1](in2))
        in3 = torch.cat([in2, out2], 1)
        out3 = self.lrelu(self.conv_list[2](in3))
        in4 = torch.cat([in3, out3], 1)
        out4 = self.lrelu(self.conv_list[3](in4))
        in5 = torch.cat([in4, out4], 1)
        out5 = self.identity(self.conv4(in5))
        
        out = out5 * 0.2 + identity

        return out

class ResRDB(nn.Module):
    """
    Multi-layer residual dense convolution block.
    """

    def __init__(self, channels: int, growths: int):
        super(ResRDB, self).__init__()
        self.rdb = nn.ModuleList([RDB(channels, growths) for i in range(3)])

    def forward(self, x: torch.Tensor):
        identity = x
        out = x

        for i in range(3):
            out = self.rdb[i](out)

        out = out * 0.2 + identity

        return out


def get_conv(in_channel, out_channel, lrelu=False):
    if lrelu:
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3,3), (1,1), (1,1)),
            nn.LeakyReLU(0.2, True)
        )
    else:
        layer = nn.Conv2d(in_channel, out_channel, (3,3), (1,1), (1,1))
    
    return layer


class Generator(nn.Module):
    """Generator network for GAN"""
    def __init__(self):
        super(Generator, self).__init__()

        fext = [ResRDB(64, 32) for i in range(16)]
        self.fext = nn.Sequential(*fext)

        self.conv1 = get_conv(3,64)
        self.conv2 = get_conv(64,64,lrelu=True)
        self.conv3 = get_conv(64,64,lrelu=True)
        self.conv4 = Involution2d(64,3)

        self.upsample = get_conv(64,64,lrelu=True)

    def forward(self, x):
        
        out1 = self.conv1(x)
        out = self.fext(out1)
        out2 = self.conv2(out)
        out = out1 + out2
        out = self.upsample(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsample(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    def __init__(self):
        super(Discriminator, self).__init__()

        channels = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
        layers = [nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
                  nn.LeakyReLU(0.2, True)]
        
        for i in range(1,len(channels)):
            layers = layers + self.make_layer(channels[i-1], channels[i], i)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100,1)
        )

    def make_layer(self, in_channel, out_channel, idx):
        kernel = [(3,3),(4,4)]
        pads = [(1,1),(2,2)]
        layer = [nn.Conv2d(in_channel, out_channel, kernel[idx%2], pads[idx%2], (1,1), bias=False),
                 nn.BatchNorm2d(out_channel),
                 nn.LeakyReLU(0.2, True)]
        return layer

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

def init_weights(gen):
    """Initialise weights for Generator appropriately."""
    for m in gen.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= 0.1
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            m.weight.data *= 0.1