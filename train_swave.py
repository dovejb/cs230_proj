import torch
from vae_spec import SpecVAE
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data import MusModule, SingleModule
from lightning_module import VocalSeparator
from convtasnet.conv_tasnet import TasNet
import os
from models.swave.swave import SWave

NAME="swave"

if __name__ == '__main__':
    #model = TasNet(num_spk=1, enc_dim=1024, feature_dim=512, stack=5)
    model = SWave(
      N=128,    #feature_dim, the channels of encoding
      L=8,      #kernel_size
      H=128,    #hidden_dim
      R=4,      #separator layer nums
      C=1,      #num spk
      sr=16000, #sample rate
      segment=100, #segment duration (ms)
      input_normalize=True,
    )
    from torchinfo import summary
    summary(model, input_size=(2,WAV_SHAPE[0]))
    tb_logger = TensorBoardLogger(save_dir="./log",
                                  name=NAME)
    trainer = Trainer(logger=tb_logger,
                      callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2,
                                        dirpath = os.path.join(tb_logger.log_dir, "checkpoints"),
                                        monitor = "val_sisnr",
                                        save_last = True,
                                        )
                      ],
                      auto_lr_find=True,
                      enable_progress_bar=True,
                      accelerator='gpu',
                      devices=-1,
                      #max_epochs=400,
                      max_epochs=-1,
                      log_every_n_steps=1,
                      gradient_clip_val=0.1,
                      auto_scale_batch_size="binsearch",
                      #resume_from_checkpoint=f"./log/{NAME}/version_2/checkpoints/last.ckpt",
                      check_val_every_n_epoch=5,
                      )
    #data = MusModule(n_train=16,n_test=16)
    data = SingleModule()
    module = VocalSeparator(model, 1e-3)

    trainer.fit(module, data)
    torch.save(module.model, f"./{NAME}.model")