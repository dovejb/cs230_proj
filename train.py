import torch
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_module import VocalSeparator
import os
import soundfile as sf
from data import *
from model.model import Model

NAME="m3"
LEN=8000

def dataproc(x,y):
    start=0
    maxlen=LEN
    def proc(x):
        x = x[...,start:start+maxlen]
        return x
    return proc(x), proc(y)

if __name__ == '__main__':
    model = Model(3)
    from torchinfo import summary
    summary(model, input_size=(16,LEN))
    #exit()
    tb_logger = TensorBoardLogger(save_dir="./log",
                                  name=NAME)
    trainer = Trainer(logger=tb_logger,
                      callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2,
                                        dirpath = os.path.join(tb_logger.log_dir, "checkpoints"),
                                        monitor = "val_loss",
                                        save_last = True,
                                        )
                      ],
                      #auto_lr_find=True,
                      enable_progress_bar=True,
                      accelerator='gpu',
                      devices=-1,
                      max_epochs=800,
                      log_every_n_steps=1,
                      gradient_clip_val=0.5,
                      #auto_scale_batch_size="binsearch",
                      check_val_every_n_epoch=100,
                      reload_dataloaders_every_n_epochs=10000,
                      )
    data = MusModule(n_train=16,n_test=1,batch_size=16,postproc=dataproc)
    module = VocalSeparator(model, 1e-2, name=NAME)

    trainer.fit(
        model=module,
        datamodule=data,
        #ckpt_path=f"./log/{NAME}/version_5/checkpoints/last.ckpt",
        )
    torch.save(module.model, f"./{NAME}.model")
