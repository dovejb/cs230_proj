import torch
from constants import *
import os
import soundfile as sf
from data import *
from model.model import Model
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm, trange

LEN=8000

def make_ckptpath(name):
    return "./ckpts/"+name+".ckpt"

def dataproc(x,y):
    start=0
    maxlen=LEN
    def proc(x):
        x = torch.Tensor(x[...,start:start+maxlen]).cuda()
        return x
    return proc(x), proc(y)

num_epochs = 10
num_train = 16
num_test = 16
batch_size = 16
base_step = 0
step = 0

trainloader = DataLoader(Muset(num_train,postproc=dataproc), batch_size, True)
testloader = DataLoader(Muset(num_test,postproc=dataproc), batch_size, False)

writer = SummaryWriter()

def write_summary(writer:SummaryWriter, step, losses):
    for name, value in losses:
        writer.add_scalar(name, value, base_step + step)
    writer.flush()

def train(
    model:Model, 
    optimizer:optim.Optimizer, 
    trainloader: DataLoader, 
    testloader: DataLoader,
):
    global step, base_step
    t = trange(num_epochs,leave=True)
    for epoch in t:
        for batchidx in range(len(trainloader.dataset) // trainloader.batch_size):
            x, y = next(iter(trainloader))
            yhat = model.forward(x)
            losses = model.loss_function(y, yhat, x=x)
            loss = losses[0][1]
            loss.backward()
            optimizer.step()

            t.set_postfix_str(f"loss:{loss:.3f} step:{base_step+step}")
            step += 1
    return loss

resume_from_checkpoint_name=""
#resume_from_checkpoint_name="1669832770"
save_checkpoint=False

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
    ).cuda()
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-3,
    )
    if resume_from_checkpoint_name != "":
        ckpt_path = "./ckpts/"+resume_from_checkpoint_name+".ckpt"
        checkpoint = torch.load(ckpt_path)
        print("Resume checkpoint", ckpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        base_step = checkpoint['step']
        print("loaded step", base_step)


    #############################################################
    #    do sth to model here (such as freeze, change lr)       #

    #############################################################

    loss = train(model, optimizer, trainloader, testloader)

    if save_checkpoint:
        t = str(int(time.time()))
        ckpt_path = make_ckptpath(t)
        torch.save({
            "step": base_step + step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        }, ckpt_path)
        print("Checkpoint saved at", t)
        print("saved_step", base_step, step)