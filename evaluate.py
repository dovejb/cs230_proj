import torch
from constants import *
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_module import VocalSeparator
import os
import soundfile as sf
from data import *
from model.model import Model
from train import *

if __name__ == '__main__':
    ckpt_path = f"./log/{NAME}/version_1/checkpoints/last.ckpt"
    ckpt_path = f"q:/cs230/mylast.ckpt" 
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    x, y = data.train_dataloader().dataset[0]
    yhat = model.forward(x)
    #sf.write("./yhat.wav", yhat.detach().cpu()[0], SR)
    print("x", torch.mean(x), torch.std(x))
    print("y", torch.mean(y), torch.std(y))
    def scale_to(yhat, x):
        my = torch.mean(yhat)
        x_scale = (torch.max(x) - torch.min(x))/2
        y_scale = (torch.max(yhat) - torch.min(yhat))/2
        yhat = (yhat-my)/y_scale*x_scale
        return yhat
    yhat = scale_to(yhat, x)
    print("yhat", torch.mean(yhat), torch.std(yhat))
    sf.write("./yhat_scaled.wav", yhat.detach().cpu()[0], SR)