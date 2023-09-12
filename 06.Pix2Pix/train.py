import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator_model import Discriminator
from generator_model import Generator
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import MultiDataset
from torch.utils.tensorboard import SummaryWriter

def train_m(disc, gen, loader, opt_disc, opt_gen, l1, bce,):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        (D_loss).backward()
        opt_disc.step()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        (G_loss).backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = MultiDataset(root_dir='facades/train', targetOnLeft=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = MultiDataset(root_dir='facades/val', targetOnLeft=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    else:
        writer = SummaryWriter("evaluation/gen/")
        x, _ = train_dataset.__getitem__(0)
        writer.add_graph(gen,x)
        writer.close()
        writer = SummaryWriter("evaluation/disc/")
        x, y = train_dataset.__getitem__(0)
        writer.add_graph(disc,(x, y))
        writer.close()
        SystemExit()

    for epoch in range(config.NUM_EPOCHS):
        train_m(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,)

        if config.SAVE_MODEL and epoch%5==0:
            save_checkpoint(gen, opt_gen, config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, config.CHECKPOINT_DISC)
        
        save_some_examples(gen, val_loader, epoch, folder='evaluation')

if __name__ == "__main__":
    main()