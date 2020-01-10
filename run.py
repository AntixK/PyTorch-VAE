from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from trainer import VAETrainer
from models import VAE


tt_logger = TestTubeLogger(
    save_dir="logs/",
    name="VanillaVAE",
    debug=False,
    create_git_tag=False
)



class hparams(object):
    def __init__(self):
        self.LR = 0.001
        self.momentum = 0.9
        self.scheduler_gamma = 0
        self.gpus = 1
        self.data_path = 'data/'
        self.batch_size = 32

hyper_params = hparams()

model = VAE(3, 32)
net = VAETrainer(model,
                 hyper_params)


trainer = Trainer(gpus=hyper_params.gpus,
                  min_nb_epochs=1,
                  max_nb_epochs=2,
                  logger=tt_logger)
trainer.fit(net)