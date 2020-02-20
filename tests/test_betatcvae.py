import torch
import unittest
from models import BetaTCVAE
from torchsummary import summary


class TestBetaTCVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = BetaTCVAE(3, 64, anneal_steps= 100)

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)

    def test_sample(self):
        self.model.cuda()
        y = self.model.sample(8, 'cuda')
        print(y.shape)

    def test_generate(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model.generate(x)
        print(y.shape)


if __name__ == '__main__':
    unittest.main()