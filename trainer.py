import torch
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
        torch.manual_seed(self.params.manual_seed)
        self.curr_device = None

    def forward(self, input: Tensor):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        real_img, _ = batch

        self.curr_device = real_img.device

        recons_img, mu, log_var = self.model(real_img)
        loss = self.model.loss_function(recons_img, real_img, mu, log_var)

        self.logger.experiment.log({'train_loss': loss.item()})

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        recons_img, mu, log_var = self.model(real_img)
        loss = self.model.loss_function(recons_img, real_img, mu, log_var)

        self.logger.experiment.log({'val_loss': loss.item()})
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        z = torch.randn(self.params.batch_size,
                        128).view(self.params.batch_size, -1, 1, 1)

        if self.on_gpu:
            z = z.cuda(self.curr_device)

        samples = self.model.decode(z).cpu()
        # print(samples.shape)
        grid = vutils.make_grid(samples, nrow=12)
        # print(grid.shape)
        self.logger.experiment.add_image(f'Samples', grid, self.current_epoch)
        vutils.save_image(samples.data, f"sample_{self.current_epoch}.png", normalize=True, nrow=12)


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