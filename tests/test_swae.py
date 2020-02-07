import torch
import unittest
from models import SWAE
from torchsummary import summary


class TestSWAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = SWAE(3, 10, reg_weight = 100)

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result)
        print(loss)


if __name__ == '__main__':
    unittest.main()