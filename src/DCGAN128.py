import torch.nn as nn

num_channels = 3
input_size = 128
output_size = 1
gen_filter = 128
dis_filter = 32


# Generator of DCGAN128
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, gen_filter * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gen_filter * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size * 16, gen_filter * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_filter * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(gen_filter * 8, gen_filter * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_filter * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(gen_filter * 4, gen_filter * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_filter * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(gen_filter * 2, gen_filter, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gen_filter),
            nn.ReLU(),
            nn.ConvTranspose2d(gen_filter, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator of DCGAN128
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, dis_filter, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_filter, dis_filter * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_filter * 2, dis_filter * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_filter * 4, dis_filter * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_filter * 8, dis_filter * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dis_filter * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dis_filter * 16, output_size, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# custom weights initialization with mean=0, stdev=0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)
