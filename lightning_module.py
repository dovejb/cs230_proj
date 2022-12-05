import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from torch import optim
from constants import *
import soundfile as sf
import time

class VocalSeparator(pl.LightningModule):
    def __init__(self,
                 model=None,
                 lr=1e-3,
                 name="",
                ):
        super(VocalSeparator, self).__init__()

        self.model = model
        self.lr = lr
        self.dumpCount = 0
        self.name = name
        self.trainstep = 0
    
    def forward(self, input: Tensor, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        x, y = batch
        yhat = self.forward(x)
        # loss should be a list<Tuple<name, loss_value>>, the 0th is loss, the others are metrics
        loss = self.model.loss_function(y, yhat, x=x) 
        self.trainstep += 1

        self.log_dict({l[0]:l[1] for l in loss})

        return loss[0][1]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        yhat = self.forward(x)
        loss = self.model.loss_function(y, yhat, x=x)
        self.log_dict({"val_"+l[0]:l[1] for l in loss})
        # dump the 0th sample
        #self.dump(yhat[0])

    def dump(self, yhat:Tensor):
        yhat = yhat.view(yhat.shape[-1])
        yhat = yhat.detach().cpu().numpy()
        self.dumpCount += 1
        sf.write(f"./out/{self.name}_{self.dumpCount}.wav", yhat, SR)

    def on_validation_end(self) -> None:
        pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr,
                               )
        optims.append(optimizer)

        #scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.7)
        #scheduler = optim.lr_scheduler.LinearLR(optimizer, 1, 0.2, 1000)
        #scheds.append(scheduler)

        return optims, scheds