import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from math import floor


def conv_out_shape(img_size):
    return floor((img_size + 2 - 3) / 2.) + 1

class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 img_size: int):
        super(EncoderBlock, self).__init__()

        # Build Encoder
        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU())

        out_size = conv_out_shape(img_size)
        self.encoder_mu = nn.Linear(out_channels * out_size ** 2 , latent_dim)
        self.encoder_var = nn.Linear(out_channels * out_size ** 2, latent_dim)

    def forward(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        h = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encoder_mu(h)
        log_var = self.encoder_var(h)

        return [result, mu, log_var]

class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int):
        super(DecoderBlock, self).__init__()

        # Build Decoder
        self.decode = nn.Sequential(nn.Linear(in_channels, in_channels),
                                    nn.BatchNorm1d(in_channels))
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_var = nn.Linear(in_channels, latent_dim)

    def forward(self, z: Tensor) -> Tensor:
        z = self.decode(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]

class LVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dims: List,
                 hidden_dims: List,
                 **kwargs) -> None:
        super(LVAE, self).__init__()

        self.latent_dims = latent_dims
        self.num_rungs = len(latent_dims)

        assert len(latent_dims) == len(hidden_dims), "Length of the latent" \
                                                     "and hidden dims must be the same"

        # Build Encoder
        modules = []
        img_size = 64
        for i, h_dim in enumerate(hidden_dims):
            modules.append(EncoderBlock(in_channels,
                                        h_dim,
                                        latent_dims[i],
                                        img_size))

            img_size = conv_out_shape(img_size)
            in_channels = h_dim

        self.encoders = nn.Sequential(*modules)

        # Build Decoder
        modules = []

        for i in range(self.num_rungs -1, 0, -1):
            modules.append(DecoderBlock(latent_dims[i],
                                        latent_dims[i-1]))

        self.decoders = nn.Sequential(*modules)

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        #
        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )

        # self.decoder = nn.Sequential(*modules)
        #
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        h = input

        # Posterior Parameters
        post_params = []
        for encoder_block in self.encoders:
            h, mu, log_var = encoder_block(h)
            post_params.append((mu, log_var))

        return post_params

    def decode(self, z: Tensor, post_params: List) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """


        for i, decoder_block in enumerate(self.decoders):
            mu_e, log_var_e = post_params[i]
            mu_t, log_var_t = decoder_block(z)
            mu, log_var = self.merge_gauss(mu_e, mu_t,
                                           log_var_e, log_var_t)
            z = self.reparameterize(mu, log_var)
            print(z.shape)


    def merge_gauss(self,
                    mu_1: Tensor,
                    mu_2: Tensor,
                    sigma_1: Tensor,
                    sigma_2: Tensor) -> List:

        p_1 = 1. / (sigma_1.exp() + 1e-7)
        p_2 = 1. / (sigma_2.exp() + 1e-7)

        mu = (mu_1 * p_1 + mu_2 * p_2)/(p_1 + p_2)
        log_var = torch.log(1./(p_1 + p_2))
        return [mu, log_var]

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
        post_params = self.encode(input)
        mu, log_var = post_params.pop()
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z, post_params)
        # return  [self.decode(z), input, mu, log_var]

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

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

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