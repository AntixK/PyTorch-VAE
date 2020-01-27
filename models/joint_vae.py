import torch
import numpy as np
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class JointVAE(BaseVAE):
    num_iter = 1

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 categorical_dim: int,
                 latent_min_capacity: float =0.,
                 latent_max_capacity: float = 25.,
                 latent_gamma: float = 30.,
                 latent_num_iter: int = 25000,
                 categorical_min_capacity: float =0.,
                 categorical_max_capacity: float = 25.,
                 categorical_gamma: float = 30.,
                 categorical_num_iter: int = 25000,
                 hidden_dims: List = None,
                 temperature: float = 0.5,
                 anneal_rate: float = 3e-5,
                 anneal_interval: int = 100, # every 100 batches
                 alpha: float = 30.,
                 **kwargs) -> None:
        super(JointVAE, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha

        self.cont_min = latent_min_capacity
        self.cont_max = latent_max_capacity

        self.disc_min = categorical_min_capacity
        self.disc_max = categorical_max_capacity

        self.cont_gamma = latent_gamma
        self.disc_gamma = categorical_gamma

        self.cont_iter = latent_num_iter
        self.disc_iter = categorical_num_iter

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
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, self.categorical_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim + self.categorical_dim,
                                       hidden_dims[-1] * 4)

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
        self.sampling_dist = torch.distributions.OneHotCategorical(1. / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.fc_z(result)
        z = z.view(-1, self.categorical_dim)
        return [mu, log_var, z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self,
                       mu: Tensor,
                       log_var: Tensor,
                       q: Tensor,
                       eps:float = 1e-7) -> Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param mu: (Tensor) mean of the latent Gaussian  [B x D]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        :param q: (Tensor) Categorical latent Codes [B x Q]
        :return: (Tensor) [B x (D + Q)]
        """

        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = e * std + mu

        # Sample from Gumbel
        u = torch.rand_like(q)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dim)

        return torch.cat([z, s], dim=1)


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, q = self.encode(input)
        z = self.reparameterize(mu, log_var, q)
        return  [self.decode(z), input, q, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]
        mu = args[3]
        log_var = args[4]

        q_p = F.softmax(q, dim=-1) # Convert the categorical codes into probabilities


        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        batch_idx = kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)

        recons_loss =F.mse_loss(recons, input, reduction='mean')

        # Adaptively increase the discrinimator capacity
        disc_curr = (self.disc_max - self.disc_min) * \
                    self.num_iter/ float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical_dim))

        # KL divergence between gumbel-softmax distribution
        eps = 1e-7

        # Entropy of the logits
        h1 = q_p * torch.log(q_p + eps)
        # Cross entropy with the categorical distribution
        h2 = q_p * np.log(1. / self.categorical_dim + eps)
        kld_disc_loss = torch.mean(torch.sum(h1 - h2, dim =1), dim=0)

        # Compute Continuous loss
        # Adaptively increase the continuous capacity
        cont_curr = (self.cont_max - self.cont_min) * \
                    self.num_iter/ float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)

        kld_cont_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                                                    dim=1),
                                   dim=0)
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kld_disc_loss) + \
                        self.cont_gamma * torch.abs(cont_curr - kld_cont_loss)
        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * capacity_loss

        if self.training:
            self.num_iter += 1
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Capacity_Loss':capacity_loss}

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
        # [S x D]
        z = torch.randn(num_samples,
                        self.latent_dim)

        M = num_samples
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M , self.categorical_dim])
        q = torch.from_numpy(np_y)

        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        z = torch.cat([z, q], dim = 1).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]