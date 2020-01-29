import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class FactorVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 gamma: float = 40.,
                 **kwargs) -> None:
        super(FactorVAE, self).__init__()

        self.latent_dim = latent_dim
        self.gamma = gamma

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

        # Discriminator network for the Total Correlation (TC) loss
        self.discriminator = nn.Sequential(nn.Linear(self.latent_dim, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 1000),
                                          nn.BatchNorm1d(1000),
                                          nn.LeakyReLU(0.2),
                                          nn.Linear(1000, 2))
        self.D_z_reserve = None


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
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]

    def permute_latent(self, z: Tensor) -> Tensor:
        """
        Permutes each of the latent codes in the batch
        :param z: [B x D]
        :return: [B x D]
        """
        B, D = z.size()

        # Returns a shuffled inds for each latent code in the batch
        inds = torch.cat([(D *i) + torch.randperm(D) for i in range(B)])
        return z.view(-1)[inds].view(B, D)

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
        mu = args[2]
        log_var = args[3]
        z = args[4]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']

        # Update the VAE
        if optimizer_idx == 0:
            recons_loss =F.mse_loss(recons, input)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            self.D_z_reserve = self.discriminator(z)
            vae_tc_loss = (self.D_z_reserve[:, 0] - self.D_z_reserve[:, 1]).mean()

            loss = recons_loss + kld_weight * kld_loss + self.gamma * vae_tc_loss

            # print(f' recons: {recons_loss}, kld: {kld_loss}, VAE_TC_loss: {vae_tc_loss}')
            return {'loss': loss,
                    'Reconstruction_Loss':recons_loss,
                    'KLD':-kld_loss,
                    'VAE_TC_Loss': vae_tc_loss}

        # Update the Discriminator
        elif optimizer_idx == 1:
            device = input.device
            true_labels = torch.ones(input.size(0), dtype= torch.long,
                                     requires_grad=False).to(device)
            false_labels = torch.zeros(input.size(0), dtype= torch.long,
                                       requires_grad=False).to(device)

            z = z.detach() # Detach so that VAE is not trained again
            z_perm = self.permute_latent(z)
            D_z_perm = self.discriminator(z_perm)
            D_tc_loss = 0.5 * (F.cross_entropy(self.D_z_reserve, false_labels) +
                               F.cross_entropy(D_z_perm, true_labels))
            # print(f'D_TC: {D_tc_loss}')
            return {'loss': D_tc_loss,
                    'D_TC_Loss':D_tc_loss}

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
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]