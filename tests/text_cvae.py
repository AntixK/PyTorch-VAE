import torch
import unittest
from models import CVAE


class TestCVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = CVAE(3, 40, 10)

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        c = torch.randn(16, 40)
        y = self.model(x, c)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)
        c = torch.randn(16, 40)
        result = self.model(x, labels = c)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()