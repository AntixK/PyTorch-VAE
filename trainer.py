import pytorch_lightning as pl
from models import BaseVAE
from torchvision import transforms
from torchvision.datasets import CelebA
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models.types_ import *


class VAETrainer(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAETrainer, self).__init__()

        self.model = vae_model
        self.params = params


    def training_step(self, batch, batch_idx):
        real_img, _ = batch
        recons_img, mu, log_var = self.model(real_img)
        loss = self.model.loss_function(recons_img, real_img, mu, log_var)

        self.logger.experiment.log({'train_loss': loss.item()})

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        recons_img, mu, log_var = self.model(real_img)
        loss = self.model.loss_function(recons_img, real_img, mu, log_var)

        self.logger.experiment.log({'val_loss': loss.item()})
        return {'loss': loss}

    def validation_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.LR)
        return [optimizer]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(CelebA(root = self.params.data_path,
                                 split = "train",
                                 transform=transforms.Compose([
                                     transforms.Resize(128),
                                     transforms.CenterCrop(128),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                      download=True),
                               batch_size= self.params.batch_size,
                               drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CelebA(root = self.params.data_path,
                                 split = "test",
                                 transform=transforms.Compose([
                                     transforms.Resize(128),
                                     transforms.CenterCrop(128),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                 download=True),
                               batch_size= self.params.batch_size,
                               drop_last=True)

    # Utils
    def save_samples(self):
        recons_img = 0
        vutils.save_image(recons_img,
                          f"{self.logger.save_dir}/fake_samples.png",
                          normalize=True)