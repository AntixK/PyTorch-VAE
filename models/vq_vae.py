import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/zalandoresearch/pytorch-vq-vae
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


# class VectorQuantizer(nn.Module):
#     """
#     Reference:
#     [1] https://github.com/zalandoresearch/pytorch-vq-vae
#     """
#     def __init__(self,
#                  num_embeddings: int,
#                  embedding_dim: int,
#                  beta: float=0.25):
#         super(VectorQuantizer, self).__init__()
#
#         self.D = embedding_dim
#         self.K = num_embeddings
#
#         self._embedding = nn.Embedding(self.K, self.D)
#         self._embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
#         self.beta = beta
#
#     def forward(self, inputs):
#         # convert inputs from BCHW -> BHWC
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape
#
#         # Flatten input
#         flat_input = inputs.view(-1, self.D)
#
#         # Calculate distances
#         distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
#                      + torch.sum(self._embedding.weight ** 2, dim=1)
#                      - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
#
#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self.K, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
#
#         # Quantize and unflatten
#         quantized_input = torch.matmul(encodings, self._embedding.weight).view(input_shape)
#
#         # Loss
#         commitment_loss = F.mse_loss(quantized_input.detach(), inputs)
#         embedding_loss = F.mse_loss(quantized_input, inputs.detach())
#         loss = embedding_loss + self.beta * commitment_loss
#
#         # quantized = inputs + (quantized - inputs).detach()
#         # avg_probs = torch.mean(encodings, dim=0)
#         # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#
#         # convert quantized from BHWC -> BCHW
#         return quantized_input.permute(0, 3, 1, 2).contiguous(), loss


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


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

        modules.append(ResidualLayer(in_channels, in_channels))

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embedding_dim),
                nn.LeakyReLU())
        )
        # modules.append(ResidualLayer(embedding_dim, embedding_dim))

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        # Build Decoder
        modules = []
        # modules.append(ResidualLayer(embedding_dim, embedding_dim))

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

        modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

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

        modules.append(nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding=1),
                            nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

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
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    # def sample(self,
    #            num_samples: int,
    #            current_device: Union[int, str], **kwargs) -> Tensor:
    #     """
    #     Samples from the latent space and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :param current_device: (Int)/(Str) Device to run the model
    #     :return: (Tensor)
    #     """
    #     # Get random encoding indices
    #     sample_inds = torch.randint(self.embedding_dim,
    #                                 (num_samples * self.img_size ** 2, 1))  # [SHW, 1]
    #
    #     # Convert to corresponding one-hot encodings
    #     sample_one_hot = torch.zeros(sample_inds.size(0), self.num_embeddings)
    #     sample_one_hot.scatter_(1, sample_inds, 1.)  # [BHW x K]
    #
    #     # Quantize the input based on the learned embeddings
    #     quantized_input = torch.matmul(sample_one_hot,
    #                                    self.vq_layer.embedding.weight.detach().cpu())  # [BHW, D]
    #     quantized_input = quantized_input.view(num_samples,
    #                                            self.img_size,
    #                                            self.img_size,
    #                                            self.embedding_dim)  # [B x H x W x D]
    #
    #     quantized_input = input + (quantized_input - input).detach()
    #     quantized_input = quantized_input.permute(0, 3, 1, 2)  # [B x D x H x W]
    #
    #     samples = self.decode(quantized_input.to(current_device))
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]