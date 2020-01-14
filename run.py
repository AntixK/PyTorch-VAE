from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from trainer import VAETrainer
from models import VAE
import torch


tt_logger = TestTubeLogger(
    save_dir="logs/",
    name="VanillaVAE",
    debug=False,
    create_git_tag=False
)

class hparams(object):
    def __init__(self):
        self.LR = 0.0005
        self.momentum = 0.9
        self.scheduler_gamma = 0
        self.gpus = 1
        self.data_path = 'data/'
        self.batch_size = 144
        self.manual_seed = 1256

hyper_params = hparams()
torch.manual_seed(hyper_params.manual_seed)
model = VAE(in_channels=3, latent_dim=128)
net = VAETrainer(model,
                 hyper_params)


trainer = Trainer(gpus=hyper_params.gpus,
                  min_nb_epochs=1,
                  max_nb_epochs=2,
                  logger=tt_logger,
                  log_save_interval=100,
                  train_percent_check=1.,
                  val_percent_check=1.)
trainer.fit(net)