import torch
from vae_spec import SpecVAE
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from mus import MusModule, SingleModule
from lightning_module import VocalSeparator
from convtasnet.conv_tasnet import TasNet
import os
from models.swave.swave import SWave
from wave_cnn import WaveCNN
import soundfile as sf

NAME="wave_cnn"
LEN=16000

if __name__ == '__main__':
    model = WaveCNN(wave_length=LEN, block_num=8, res_kernel=7, res_loop_count=1, initial_n_chan=2048)
    from torchinfo import summary
    summary(model, input_size=(2,1,LEN))
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
                      auto_lr_find=True,
                      enable_progress_bar=True,
                      accelerator='gpu',
                      devices=-1,
                      max_epochs=800,
                      log_every_n_steps=1,
                      gradient_clip_val=0.1,
                      auto_scale_batch_size="binsearch",
                      #resume_from_checkpoint=f"./log/{NAME}/version_2/checkpoints/last.ckpt",
                      check_val_every_n_epoch=100,
                      )
    #data = MusModule(n_train=16,n_test=16)
    def dataproc(x,y):
        start=20000
        maxlen=LEN
        def proc(x):
            x = x[...,start:start+maxlen]
            x = x.reshape(1,-1)
            return x
        return proc(x), proc(y)
    data = SingleModule(postproc=dataproc)
    y = data.dataset[0][1].reshape(-1)
    sf.write(f"./out/{NAME}_y.wav", y, SR)
    module = VocalSeparator(model, 1e-3, name=NAME)

    trainer.fit(module, data)
    torch.save(module.model, f"./{NAME}.model")
