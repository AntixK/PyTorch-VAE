import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math

class IWAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 num_samples: int = 5,
                 **kwargs) -> None:
        super(IWAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_samples = num_samples

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [B x S x D]
        :return: (Tensor) [B x S x C x H x W]
        """
        B, _, _ = z.size()
        z = z.view(-1, self.latent_dim) #[BS x D]
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result) #[BS x C x H x W ]
        result = result.view([B, -1, result.size(1), result.size(2), result.size(3)]) #[B x S x C x H x W]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        z= self.reparameterize(mu, log_var) # [B x S x D]
        eps = (z - mu) / log_var # Prior samples
        return  [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]

        input = input.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) #[B x S x C x H x W]
        _, _, c, h, w = input.shape
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        log_p_x_z = -((recons - input) ** 2).flatten(2).mean(-1) # Reconstruction Loss [B x S]
        log_p_z = (- 0.5 * ((z - 0.0) ** 2 / 1.0) - 0.5 * torch.log(2.0 * math.pi * 1.0)).mean(-1) # prior [B x S]
        log_q_z_x = (- 0.5 * ((z - mu) ** 2 / log_var.exp()) - 0.5 * (torch.log(2.0 * math.pi) + log_var)).mean(-1) # variational posterior [B x S]
        # this kl divergence is estimated, actually its kind of weird to write it in a beta-VAE way
        # probably it is better to do \lambda * reconstruction loss + DKL rather than reconstruction loss + * \beta * DKL here
        # I am just trying to make it maintains the same meaning as before ...
        # Infact it is SGBV 1 (Auto-encoding Variational Bayes Eq 6) used here, not SGVB 2 (Auto-encoding Variational Bayes Eq 7, Eq 10)
        kld_loss_est = kld_weight * (log_q_z_x - log_p_z) * (c * h * w) # [B x S]

        # kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2) ## [B x S]
        # Get importance weights, needs to be detached
        elbo = (log_p_x_z - kld_loss_est)

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(elbo.detach(), dim = -1)

        loss = - torch.mean(torch.sum(weight * elbo, dim=-1), dim = 0)

        return {'loss': loss, 'Reconstruction_Loss':log_p_x_z.mean(), 'KLD':-kld_loss_est.mean()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0][:, 0, :]
