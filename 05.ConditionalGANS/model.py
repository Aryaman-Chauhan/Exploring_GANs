''' 
    https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
    https://arxiv.org/abs/1701.07875 : WGAN Paper
    https://arxiv.org/abs/1704.00028 : WGAN-GP Paper
    Implemented the model in the model.py file, and training will be done in train.py file
    Implementing Discriminator and Generator according to the paper
    Actually, this is the same model as we used for DCGAN, except one change, that the discriminator here
    is called Critic, and doesn't have sigmoid at the end

    Now, we will build a conditional GAN, where the model will try to generate data based on label. This 
    will allow us to decide what the output should be. This will allow us to have more control over our 
    model and help us get a more sensible model.
    NOTE: Our data will now map from discriminator to Generator
'''

import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        self.img_size = img_size
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channel_img x 64 x 64
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),# 32x32 +1 for labels
                # So now, the additional channel contains the detail/label of the image
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),# 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1),# 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1),# 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),# 1x1
            )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

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
                    nn.InstanceNorm2d(out_channels, affine=True), # Different from WGAN
                    nn.LeakyReLU(0.2), # Following the paper
            )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        self.img_size = img_size
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x Z_dim x 1x1
            self._block(z_dim + embed_size, features_g*16, 4, 1, 0), # N x f_g*16 x 4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # [-1, 1], we're going to normalize the image in this range
        )
        self.embed = nn.Embedding(num_classes, embed_size) #Not added to image but to noise

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
    
    def forward(self, x, labels):
        # latent noise x: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)
    
def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0. , 0.02) # Standards by paper

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success')

# test()