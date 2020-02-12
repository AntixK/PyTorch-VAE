import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(1./self.K, 1./self.K)

    def forward(self, input: Tensor):
        input = input.permute(0, 2, 3, 1) # [B x D x H x W] -> [B x H x W x D]
        input_shape = input.shape
        flat_input = input.contiguous().view(-1, self.D)    # [BHW x D]

        # Compute L2 distance between input and embedding weights
        dist = torch.sum(flat_input**2, dim = 1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim = 1) - \
               2 * torch.matmul(flat_input, self.embedding.weight.t()) # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim = 1).view(-1, 1) # [BHW, 1]

        # Convert to one-hot encodings
        device = input.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1.) # [BHW x K]

        # Quantize the input
        quantized_input = torch.matmul(encoding_one_hot, self.embedding.weight) # [BHW, D]
        quantized_input = quantized_input.view(input_shape) # [B x H x W x D]

        return quantized_input.permute(0, 3, 1, 2) # [B x D x H x W]


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

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

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embedding_dim),
                nn.LeakyReLU())
        )
        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(embedding_dim,
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU())
        )

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
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
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
        encoding = self.encode(input)[0]
        quantized_inputs = self.vq_layer(encoding)
        return  [self.decode(quantized_inputs), input, quantized_inputs, encoding]

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
        e = args[2]
        Z_e = args[3]

        recons_loss =F.mse_loss(recons, input)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(e.detach(), Z_e)
        embedding_loss = F.mse_loss(e, Z_e.detach())

        loss = recons_loss + embedding_loss + self.beta * commitment_loss
        return {'loss': loss,
                'Reconstruction_Loss':recons_loss,
                'Embedding_Loss':embedding_loss,
                'Commitment_Loss':commitment_loss}

    def sample(self,
               num_samples:int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int)/(Str) Device to run the model
        :return: (Tensor)
        """
        # Get random encoding indices
        sample_inds = torch.randint(self.embedding_dim,
                                    (num_samples * self.img_size ** 2, 1),
                                    device = current_device)  # [SHW, 1]

        # Convert to corresponding one-hot encodings
        sample_one_hot = torch.zeros(sample_inds.size(0), self.num_embeddings).to(current_device)
        sample_one_hot.scatter_(1, sample_inds, 1.)  # [BHW x K]

        # Quantize the input based on the learned embeddings
        quantized_input = torch.matmul(sample_one_hot,
                                       self.vq_layer.embedding.weight)  # [BHW, D]
        quantized_input = quantized_input.view(num_samples,
                                               self.img_size,
                                               self.img_size,
                                               self.embedding_dim)  # [B x H x W x D]

        quantized_input = quantized_input.permute(0, 3, 1, 2)  # [B x D x H x W]

        samples = self.decode(quantized_input)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]