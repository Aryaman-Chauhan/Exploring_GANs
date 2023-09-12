import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2): #Discriminator generally has a stride of 2
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, bias=False, padding_mode='reflect'),
            # Padding mode to reflect reduced artifacts according to paper
            nn.BatchNorm2d(out_channels), # Replacing BatchNorm with InstanceNorm will help with artifacts
            nn.LeakyReLU(0.2),# Discriminator sticks to LeakyReLU, similar to other GANS
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 256 -> 30x30
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            # Times 2 because we send in both the image and the target image
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, 4, stride=1, padding=1, padding_mode='reflect'),
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        x = self.initial(x)
        return self.model(x)
    
def test():
    x= torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)
    # print(model)
    writer = SummaryWriter("evaluation/disc/")
    writer.add_graph(model,(x, y))
    writer.close()

if __name__ == '__main__':
    test()