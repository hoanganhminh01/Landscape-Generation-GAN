import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import DCGAN128
from DCGAN128 import weights_init
from torchvision.transforms import transforms
from IPython.display import HTML
from matplotlib import animation
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)


def train(generator, discriminator, train_loader, optimizer_gen, optimizer_dis, criterion, noise_size):
    img_list = []
    generator_losses = []
    discriminator_losses = []
    iters = 0
    num_epochs = 1000
    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(64, noise_size, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, data in tqdm(enumerate(train_loader, 0)):
            discriminator.zero_grad()
            real_input = data[0].to(device)
            batch_size = real_input.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_input).view(-1)
            discriminator_loss_real = criterion(output, label)
            discriminator_loss_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            discriminator_loss_fake = criterion(output, label)
            discriminator_loss_fake.backward()
            D_G_z1 = output.mean().item()
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            optimizer_dis.step()

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            generator_loss = criterion(output, label)
            generator_loss.backward()
            D_G_z2 = output.mean().item()
            optimizer_gen.step()

            if i % 15 == 0:
                print('[%d/%d][%d/%d]\tDiscriminator Loss: %.4f\tGenerator Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         discriminator_loss.item(), generator_loss.item(), D_x, D_G_z1, D_G_z2))

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            if (iters % 250 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=1, normalize=True))
            iters += 1
    plot_loss(generator_losses, discriminator_losses)
    plot_images(train_loader, img_list)


def plot_loss(generator_losses, discriminator_losses):
    plt.figure(figsize=(10, 5))
    plt.title("DCGAN128 Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="Generator")
    plt.plot(discriminator_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../outputs/loss128.png")
    plt.show()


def plot_images(train_loader, img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images of DCGAN128")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    f = r"d:/Landscape-Generation-GAN/outputs/animation128.gif"
    writergif = animation.PillowWriter(fps=5)
    ani.save(f, writer=writergif)

    # Plot real vs. fake images
    real_batch = next(iter(train_loader))
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=1, normalize=True).cpu(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("DCGAN128 Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig("../outputs/image128.png")
    plt.show()


def main():
    image_size = 128
    batch_size = 128
    noise_size = 128
    lr = 0.0002
    dataroot = '../data_preprocessed_128/'
    generator = DCGAN128.Generator().to(device)
    discriminator = DCGAN128.Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    train(generator, discriminator, train_loader, optimizer_gen, optimizer_dis, criterion, noise_size)


if __name__ == "__main__":
    main()
