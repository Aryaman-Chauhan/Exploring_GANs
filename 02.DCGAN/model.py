''' https://arxiv.org/abs/1511.06434 : DCGAN Paper
    Implemented the model in the model.py file, and training will be done in train.py file
    Implementing Discriminator and Generator according to the paper
'''

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channel_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),# 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),# 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1),# 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1),# 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),# 1x1
            nn.Sigmoid(), #Output 0 or 1
            )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # DCGAN follows a very nice structure that they have a CNN block, batch norm. and a leaky ReLU, so we utilize this
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False # Because we are using Batch Norm.
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2), # Following the paper
            )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x Z_dim x 1x1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # [-1, 1], we're going to normalize the image in this range
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0. , 0.02) # Standards by paper

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success')

# test()