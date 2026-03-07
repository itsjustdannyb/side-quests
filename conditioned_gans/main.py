import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam

import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

import argparse
import matplotlib.pyplot as plt

from models import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")


parser = argparse.ArgumentParser(description="train the Generative Adversarial network!")
parser.add_argument("-lr", "--learning_rate", type=float, metavar="", required=False, default=1e-3, help="learning rate for the optimizers")
parser.add_argument("--epochs", type=int, metavar="", required=False, default=10, help="number of epocsh to train")
parser.add_argument("-bs", "--batch_size", type=int, metavar="", required=False, default=32, help="the size of each batch of the dataset")
parser.add_argument("--latent_dim", type=int, metavar="", required=False, default=100, help="the size of the latenet_space")


# parse args
args = parser.parse_args()

def train_gan(epochs, lr, batch_size, latent_dim):

    transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MNIST(root="data/train", train=True, transform=transforms, download=True)
    val_dataset = MNIST(root="data/val", train=False, transform=transforms, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    generator = Generator(latnet_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    generator_optimizer = Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr)


    for epoch in range(epochs):
        total_gen_loss, total_disc_loss = 0, 0
        for images, _ in train_dataloader:
            images = images.to(device)
            current_batch_size = images.shape[0]

            generator.train(); discriminator.train()


            ## train discriminator
            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_images = generator(z)

            disc_real_preds = discriminator(images)
            disc_fake_preds = discriminator(fake_images.detach()) # detach so we don't backprop into generator during disc step

            disc_real_loss = criterion(disc_real_preds.squeeze(dim=1), torch.ones(current_batch_size, device=device))
            disc_fake_loss = criterion(disc_fake_preds.squeeze(dim=1), torch.zeros(current_batch_size, device=device))

            loss_disc = (disc_real_loss + disc_fake_loss) / 2

            discriminator_optimizer.zero_grad() # reset gradients
            loss_disc.backward()
            discriminator_optimizer.step()

            ## train generator    

            fake_images_preds = discriminator(fake_images)
            loss_gen = criterion(fake_images_preds.squeeze(dim=1), torch.ones(current_batch_size, device=device)) # compute loss so gen knows if its generating well or not; the generetor never sees what a real image looks like

            generator_optimizer.zero_grad()
            loss_gen.backward()
            generator_optimizer.step()

            total_disc_loss += loss_disc.item()
            total_gen_loss += loss_gen.item()

        total_disc_loss /= len(train_dataloader)
        total_gen_loss /= len(train_dataloader)

        print(f"{epoch+1}/{epochs} | D_loss: {total_disc_loss:.4f} | G_loss: {total_gen_loss:.4f}")


    # save model
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("models saved sucessfully!s")



if __name__ == "__main__":
    train_gan(epochs=args.epochs, lr=args.learning_rate, batch_size=args.batch_size, latent_dim=args.latent_dim )