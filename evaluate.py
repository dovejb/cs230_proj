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

def evaluate_local_mask():
    ckpt_path = f"./ckpts/local_mask.ckpt"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    x, y = next(iter(data.train_dataloader()))
    x, y = x.cuda(), y.cuda()
    print(x.shape, y.shape)
    yhat = model.forward(x)
    wlen = 16000
    wnd = 512
    hop = 160
    for i in range(wlen//hop):
        start = hop * i
        end = start + wnd
        if end > wlen:
            end = wlen
        yi = y[:,start:end]
        yhati = yhat[:,start:end]
        loss = -torch.mean(metrics.sisnr(yi, yhati))
        print(loss.detach().cpu().numpy())

if __name__ == '__main__':
    if True:
        evaluate_local_mask()
        exit()
    ckpt_path = f"./ckpts/10song-40-1202.ckpt"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    x, y = data.train_dataloader().dataset[1]
    x, y = x.cuda(), y.cuda()
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
    sf.write("./x.wav", x.cpu(), SR)
    sf.write("./y.wav", y.cpu(), SR)
    sf.write("./yhat_scaled.wav", yhat.detach().cpu()[0], SR)