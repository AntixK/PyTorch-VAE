import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels= 2*latent_dim,
                          kernel_size=3, stride=1, padding  = 1),
                nn.BatchNorm2d(2*latent_dim),
                nn.ReLU())
        )

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        in_channels = latent_dim

        for _ in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=64,
                              kernel_size= 3, padding= 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True))
            )
            in_channels = 64

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(64, out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = result[:, :self.latent_dim, :, :]
        log_var = result[:, self.latent_dim:, :, :]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var

    def loss_function(self,
                      recons: Tensor,
                      input: Tensor,
                      mu: Tensor,
                      log_var: Tensor) -> Tensor:

        recons_loss =F.mse_loss(recons,
                                input,
                                reduction='mean')


        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        kld_loss /= input.size(0)
        return recons_loss + kld_loss


