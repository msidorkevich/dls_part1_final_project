import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_size):
        super().__init__()

        input_channels = input_shape[0]
        input_size = input_shape[1]

        target_size = input_size // 4

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * target_size * target_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128 * target_size * target_size),
            nn.BatchNorm1d(128 * target_size * target_size),
            nn.ReLU(True),
            nn.Unflatten(1, (128, target_size, target_size)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        latent_code = latent.clone().detach()

        return reconstruction, latent_code, latent_code
