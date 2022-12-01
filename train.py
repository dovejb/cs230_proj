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
import librosa

NAME="m1"

hyperparameters = {
    #embedding parameters
    "embed_type":3,  # 1-STFT 2-WAVE 3-CONV
    "wave_length":32000,
    "window_size":512,
    "hop_size":160,
    "num_features":512,
    #encoder parameters
    "num_heads":16,
    "local_size":8,
    "dropout":0.0,
    #tcn parameters
    "tcn_bottleneck_dim": 128,
    "num_tcn_block_channels":512,
    "num_dilations":4,
    "num_repeats":3,
}
other_configs = {
    "num_epochs": 1000,
    "num_train": 16,#18432,
    "num_test": 16,#256,
    "batch_size": 1,
    "lr": 1e-4,
    "freeze_layers": [],#[0,2,3]
    "train_fnum": 5,
}

LEN=hyperparameters["wave_length"]

def dataproc(x,y):
    start=0
    maxlen=LEN
    def proc(x):
        x = x[...,start:start+maxlen]
        x = torch.Tensor(x).cuda()
        return x
    x, y = proc(x), proc(y)
    return x, y

def wavdb(x:torch.Tensor):
    x/torch.max(x)
def fastdump(x:torch.Tensor,y:torch.Tensor, label=""):
    x, y = x.cpu().numpy(), y.cpu().numpy()
    print(label, "fast dumping x", f"max:{np.max(x)} db:{np.max(librosa.amplitude_to_db(x))}")
    sf.write("./x.wav", x, SR)
    print(label, "fast dumping y", f"max:{np.max(y)} db:{np.max(librosa.amplitude_to_db(y))}")
    sf.write("./y.wav", y, SR)

model = Model(**hyperparameters)
data = DataModule(
    n_train=other_configs["num_train"],
    n_test=other_configs["num_test"],
    batch_size=other_configs["batch_size"],
    seglen=hyperparameters["wave_length"],
    fnum=other_configs["train_fnum"],
)
module = VocalSeparator(model, other_configs["lr"], name=NAME)
if __name__ == '__main__':
    model.freeze_layers(other_configs["freeze_layers"])
    if False:
        from torchinfo import summary
        summary(model, input_size=(16,LEN))
        exit()
    if False:
        for i in range(1):
            x, y = data.train_dataloader().dataset[i]
            fastdump(x, y, str(i))
        exit()
    tb_logger = TensorBoardLogger(save_dir="./log",
                                  name=NAME)
    trainer = Trainer(logger=tb_logger,
                      callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=0,
                                        dirpath = os.path.join(tb_logger.log_dir, "checkpoints"),
                                        monitor = "val_loss",
                                        save_last = True,
                                        ),
                      ],
                      auto_lr_find=True,
                      enable_progress_bar=True,
                      accelerator='gpu',
                      devices=-1,
                      max_epochs=other_configs["num_epochs"],
                      log_every_n_steps=1,
                      #gradient_clip_val=0.5,
                      check_val_every_n_epoch=1000,
                      )
    trainer.fit(
        model=module,
        datamodule=data,
        #ckpt_path=f"./log/{NAME}/version_1/checkpoints/last.ckpt",
        )
    torch.save({
        "model": model.state_dict(),
        "hyperparameters": hyperparameters,
        "other_configs": other_configs,
    }, "q:/cs230/mylast.ckpt")