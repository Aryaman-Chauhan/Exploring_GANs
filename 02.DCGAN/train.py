'''
We will import and train Discriminator and Generator from model.py file.
We will first do this on MNIST dataset.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]), # Done this so that we don't have to modify the code when there are three channels
    ]
)

# dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms, download=False) # For MNIST dataset
# dataset = datasets.ImageFolder(root='celeb_dataset', transform=transforms) # For celeb dataset
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device=device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device=device)
initialize_weights(disc)
initialize_weights(gen)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        fake = gen(noise)
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc.zero_grad()
        loss_disc = (loss_disc_fake + loss_disc_real)/2
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

'''
OUTPUT: here, loss doesn't make much sense
Epoch [0/10] Batch 0/469                 Loss D: 0.6958, loss G: 0.7802
Epoch [0/10] Batch 100/469                 Loss D: 0.0150, loss G: 4.1284
Epoch [0/10] Batch 200/469                 Loss D: 0.5609, loss G: 1.0070
Epoch [0/10] Batch 300/469                 Loss D: 0.5233, loss G: 1.1645
Epoch [0/10] Batch 400/469                 Loss D: 0.6174, loss G: 1.0874
Epoch [1/10] Batch 0/469                 Loss D: 0.6400, loss G: 1.2121
'''