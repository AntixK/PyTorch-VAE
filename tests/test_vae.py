import torch
import unittest
from models import VanillaVAE, VAE
from torchsummary import summary


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = VanillaVAE(3, 10)

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 128, 128)

        y, m, s = self.model(x)
        loss = self.model.loss_function(y, x, m, s)
        print(loss)


if __name__ == '__main__':
    unittest.main()

# import torch.nn as nn
# import torch
#
# a = nn.ConvTranspose2d(3, 16, 3, stride = 2, padding=1, output_padding=1)
# b = nn.Conv2d(3, 16, 3)
# c = nn.Upsample(scale_factor=2, mode='bilinear',
#                                align_corners=True)
# x = torch.randn(10,3,28,28)
#
# print(a.weight.shape, b.weight.shape)
# print(a(x).shape, b(x).shape, c(x).shape)