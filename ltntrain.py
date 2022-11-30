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

NAME="m1"
LEN=8000

def dataproc(x,y):
    start=0
    maxlen=LEN
    def proc(x):
        x = x[...,start:start+maxlen]
        return x
    return proc(x), proc(y)

if __name__ == '__main__':
    model = Model(
        #embedding parameters
        embed_type=3,  # 1-STFT 2-WAVE 3-CONV
        wave_length=8000,
        window_size=512,
        hop_size=160,
        num_features=512, 
        #encoder parameters
        num_mha_layers=8,
        ffn_dim=128,
        num_heads=16,
        local_size=1,
        dropout=0.2,
        #tcn parameters
        num_tcn_block_channels=2048,
        num_dilations=4,
        num_repeats=4,
    )
    model.freeze_all()
    if False:
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
                                        ),
                      ],
                      enable_progress_bar=True,
                      accelerator='gpu',
                      devices=-1,
                      max_epochs=1000,
                      log_every_n_steps=1,
                      #gradient_clip_val=0.5,
                      check_val_every_n_epoch=500,
                      )
    data = MusModule(n_train=16,n_test=16,batch_size=16,postproc=dataproc)
    module = VocalSeparator(model, 5e-3, name=NAME)

    trainer.fit(
        model=module,
        datamodule=data,
        ckpt_path=f"./log/{NAME}/version_0/checkpoints/last.ckpt",
        )
    torch.save(module.model, f"./{NAME}.model")
