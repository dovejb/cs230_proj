import torch
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_module import VocalSeparator
import os
from wave_cnn import WaveCNN
import soundfile as sf
from data import *

print("Entering train_wave_cnn.py")

NAME="wave_cnn_full"
LEN=4000

def dataproc(x,y):
    start=0
    maxlen=LEN
    def proc(x):
        x = x[...,start:start+maxlen]
        x = x.reshape(1,-1)
        return x
    return proc(x), proc(y)

if __name__ == '__main__':
    model = WaveCNN(wave_length=LEN, block_num=8, res_kernel=7, res_loop_count=1, initial_n_chan=1024, chan_reduce_rate=2)
    from torchinfo import summary
    summary(model, input_size=(2,1,LEN))
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
                      check_val_every_n_epoch=10,
                      reload_dataloaders_every_n_epochs=10000,
                      )
    #data = SingleModule(postproc=dataproc)
    #data = MusModule(n_train=16,n_test=16)
    #y = data.dataset[0][1].reshape(-1)
    #sf.write(f"./out/{NAME}_y.wav", y, SR)
    data = MusModule(n_train=16,n_test=1,batch_size=16,postproc=dataproc)
    module = VocalSeparator(model, 1e-2, name=NAME)

    trainer.fit(
        model=module,
        datamodule=data,
        #ckpt_path=f"./log/{NAME}/version_5/checkpoints/last.ckpt",
        )
    torch.save(module.model, f"./{NAME}.model")
