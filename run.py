import torch
from models import VanillaVAE
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger


tt_logger = TestTubeLogger(
    save_dir="logs/",
    name="VanillaVAE",
    debug=False,
    create_git_tag=False,
)


class hparams(object):
    def __init__(self):
        self.LR = 5e-3
        self.scheduler_gamma = 0.95
        self.gpus = 1
        self.data_path = "../../shared/Data/"
        self.batch_size = 144
        self.img_size = 64
        self.manual_seed = 1256

hyper_params = hparams()
torch.manual_seed = hyper_params.manual_seed
model = VanillaVAE(in_channels=3, latent_dim=128)
experiment = VAEXperiment(model,
                          hyper_params)


runner = Trainer(gpus=hyper_params.gpus,
                 default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 max_nb_epochs= 50,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.)

runner.fit(experiment)