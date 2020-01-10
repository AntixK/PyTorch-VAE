import torch
import unittest
from models import VanillaVAE


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = VanillaVAE(3, 10)

    def test_forward(self):
        x = torch.randn(16, 3, 128, 128)
        y = self.model(x)
        print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 128, 128)

        y, m, s = self.model(x)
        loss = self.model.loss_function(y, x, m, s)
        print(loss)


if __name__ == '__main__':
    unittest.main()