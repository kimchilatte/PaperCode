from pathlib import Path
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        r"""
        inputs: sample images
        """
        inputs = inputs.view(-1, 28*28)
        return self.layers(inputs)


class Generator(nn.Module):
    """Generator"""
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh()
        )
    
    def forward(self, inputs):
        r"""
        inputs: random values from prior, z
        """
        return self.layers(inputs).view(-1, 1, 28, 28)


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        r"""
        Code Reference: PyTorch-Lightining https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=ArrPXFM371jR
        """
        super().__init__()
        self.hparams = hparams
        self.D = Discriminator()
        self.G = Generator(self.hparams.latent_dim)
        # cache
        self.generated_imgs = None
        self.last_imgs = None
        
    def forward(self, z):
        return self.G(z)

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        bs = x.size(0)
        self.last_imgs = x
        # training G
        if optimizer_idx == 0:  
            z = torch.randn(bs, self.hparams.latent_dim)
            if self.on_gpu:
                z = z.to(x.device)
            self.generated_imgs = self(z)

            valid_label = torch.ones(bs, 1)
            if self.on_gpu:
                valid_label = valid_label.to(x.device)

            # Pass the generated inputs to Discriminator
            # Let the Generator learn how to make a good faked images
            g_loss = self.criterion(self.D(self.generated_imgs), valid_label)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # training D
        if optimizer_idx == 1:
            valid_label = torch.ones(bs, 1)
            if self.on_gpu:
                valid_label = valid_label.to(x.device)

            # Pass the real inputs to Discriminator
            # Let the Discriminator learn to determine whether inputs are real 
            real_loss = self.criterion(self.D(x), valid_label)

            fake_label = torch.zeros(bs, 1)
            if self.on_gpu:
                fake_label = fake_label.to(x.device)

            # Pass the cached generated image
            fake_loss = self.criterion(self.D(self.generated_imgs.detach()), fake_label)

            # Discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        G_optimizer = optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        D_optimizer = optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))
        return [G_optimizer, D_optimizer], []

    def train_dataloader(self):
        root_dir = Path(".").absolute().parent.parent
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(
            root_dir / "data", 
            train=True, 
            download=True, 
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)
        return loader
