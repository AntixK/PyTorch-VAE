import torch
import unittest
from models import VanillaVAE
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
        x = torch.randn(16, 3, 64, 64)
        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)
        
    def test_reconstruction(self):
        x = torch.randn(1, 3, 64, 64)
        reconstructed, _, _ = self.model(x)
        mse = torch.nn.functional.mse_loss(reconstructed, x)
        print(f"Reconstruction MSE: {mse.item()}")

if __name__ == '__main__':
    unittest.main()
