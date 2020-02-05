import torch
import unittest
from models import LVAE
from torchsummary import summary


class TestLVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = LVAE(3, [4,8,16,32,128], hidden_dims=[32, 64,128, 256, 512])

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        # print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()