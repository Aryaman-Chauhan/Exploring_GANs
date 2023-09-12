'''
We will import and train Discriminator and Generator from model.py file.
We will first do this on MNIST dataset.
WGAN Provides two benefits:
    - Stability 
    - Loss make sense
They achieve this by using Wasserstein Loss instead of Shannon Loss
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
# WEIGHT_CLIP = 0.01 # Removed for WGAN-GP
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]), # Done this so that we don't have to modify the code when there are three channels
    ]
)

# dataset = datasets.MNIST(root='D:/Exploring_GANs/02.DCGAN/dataset/', train=True, transform=transforms, download=False) # For MNIST dataset
dataset = datasets.ImageFolder(root='D:/Exploring_GANs/02.DCGAN/celeb_dataset', transform=transforms) # For celeb dataset
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device=device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device=device) # For WGAN, discriminator is stated as critic
initialize_weights(critic)
initialize_weights(gen)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)
        ### Train Critic: min E(critic(x)) - E(critic(gen(z)))
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic=critic, labels=labels, real=real, fake=fake, device=device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) 
                            + LAMBDA_GP * gp) # We want to minimize, so -ve sign
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        ### Train Generator: min E(critic(gen_fake))
        output = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels) # Changed from fixed_noise to noise so labels match
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1