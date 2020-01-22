import torch
import unittest
from models import FactorVAE
from torchsummary import summary


class TestFAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = FactorVAE(3, 10)

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cpu'))
        #
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())

        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)
        x2 = torch.randn(16,3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005, optimizer_idx=0, secondary_input=x2)
        loss = self.model.loss_function(*result, M_N = 0.005, optimizer_idx=1, secondary_input=x2)
        print(loss)

    def test_optim(self):
        optim1 = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        optim2 = torch.optim.Adam(self.model.discrminator.parameters(), lr = 0.001)

    def test_sample(self):
        self.model.cuda()
        y = self.model.sample(144, 0)




if __name__ == '__main__':
    unittest.main()