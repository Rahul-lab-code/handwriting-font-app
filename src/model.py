# filepath: handwriting-font-app/handwriting-font-app/src/model.py

import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=26):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 16 * 16 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16 + num_classes, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim + num_classes, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        x = self.encoder(x)
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes).float()
        x = torch.cat([x, y_onehot], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes).float()
        z = torch.cat([z, y_onehot], dim=1)
        x = self.fc_decode(z)
        x = x.view(-1, 64, 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.randint(0, self.num_classes, (num_samples,)).to(device)
        return self.decode(z, labels)