import torch
import unittest
from models import JointVAE
from torchsummary import summary


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = JointVAE(3, 10, 40, 0.0)

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(128, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005, batch_idx=5)
        print(loss)


    def test_sample(self):
        self.model.cuda()
        y = self.model.sample(144, 0)
        print(y.shape)


if __name__ == '__main__':
    unittest.main()