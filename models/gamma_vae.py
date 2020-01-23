import torch
from models import BaseVAE
from torch import nn
from torch.distributions import Gamma
from torch.nn import functional as F
from .types_ import *


class GammaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 gamma_shape: float = 8.,
                 prior_shape: float = 2.,
                 prior_rate: float = 1.,
                 **kwargs) -> None:
        super(GammaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.B = gamma_shape

        self.prior_alpha = torch.tensor([prior_shape])
        self.prior_beta = torch.tensor([prior_rate])

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

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
                                       stride=2,
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
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
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
        alpha = F.softplus(self.fc_mu(result))
        beta = F.softplus(self.fc_var(result))

        return [alpha, beta]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, alpha: Tensor, beta: Tensor) -> Tensor:
        """
        Reparameterize the Gamma distribution by the shape augmentation trick.
        Reference:
        [1] https://arxiv.org/pdf/1610.05683.pdf

        :param alpha: (Tensor) Shape parameter of the latent Gamma
        :param beta: (Tensor) Rate parameter of the latent Gamma
        :return:
        """
        # Sample from Gamma to guarantee acceptance
        alpha_ = alpha.clone().detach()
        z_hat = Gamma(alpha_ + self.B, 1).sample()

        # Compute the eps ~ N(0,1) that produces z_hat
        eps = self.inv_h_func(alpha + self.B , z_hat)
        z = self.h_func(alpha + self.B, eps)

        # When beta != 1, scale by beta
        return z / beta

    def h_func(self, alpha: Tensor, eps: Tensor) -> Tensor:
        """
        Reparameterize a sample eps ~ N(0, 1) so that h(z) ~ Gamma(alpha, 1)
        :param alpha: (Tensor) Shape parameter
        :param eps: (Tensor) Random sample to reparameterize
        :return: (Tensor)
        """

        z = (alpha - 1./3.) * (1 + eps / (torch.sqrt(9. * alpha - 3.)))**3
        return z

    def inv_h_func(self, alpha: Tensor, z: Tensor) -> Tensor:
        """
        Inverse reparameterize the given z into eps.
        :param alpha: (Tensor)
        :param z: (Tensor)
        :return: (Tensor)
        """
        eps = torch.sqrt(9. * alpha - 3.) * ((z / (alpha - 1./3.))**(1. / 3.) - 1.)
        return eps

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        alpha, beta = self.encode(input)
        z = self.reparameterize(alpha, beta)

        return [self.decode(z), input, alpha, beta]

    def I_function(self, alpha_p, beta_p, alpha_q, beta_q):
        return - (alpha_q * beta_q) / alpha_p - \
               beta_p * torch.log(alpha_p) - torch.lgamma(beta_p) + \
               (beta_p - 1) * torch.digamma(beta_q) + \
               (beta_p - 1) * torch.log(alpha_q)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        alpha = args[2]
        beta = args[3]

        curr_device = input.device
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        # https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
        alpha = 1./ alpha
        beta = 1./ beta

        self.prior_alpha = self.prior_alpha.to(curr_device)
        self.prior_beta = self.prior_beta.to(curr_device)

        kld_loss = self.I_function(self.prior_alpha, self.prior_beta, self.prior_alpha, self.prior_beta) - \
                   self.I_function(alpha, beta, self.prior_alpha, self.prior_beta)

        kld_loss = torch.mean(torch.sum(kld_loss, dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = Gamma(self.prior_alpha, self.prior_beta).sample((num_samples, self.latent_dim))
        z = z.squeeze().to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
