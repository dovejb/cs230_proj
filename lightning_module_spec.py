import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, Tensor
from torch import optim
from constants import *

class VocalSeparatorSpec(pl.LightningModule):
    def __init__(self,
                 model,
                 lr,
                ):
        super(VocalSeparatorSpec, self).__init__()

        self.model = model
        self.lr = lr
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        x, y = batch
        yhat = self.forward(x)
        loss = self.model.loss_function(yhat, y)
        """
        train_loss = self.model.loss_function(y,
                                            *results, 
                                            optimizer_idx=optimizer_idx, 
                                            batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        """
        self.log_dict({"loss":loss})

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        yhat = self.forward(x)
        loss = self.model.loss_function(yhat, y)
        self.log_dict({"val_loss": loss})
        """
        results = self.forward(x)
        val_loss = self.model.loss_function(y, 
                                            *results, 
                                            optimizer_idx=optimizer_idx, 
                                            batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        """
    def on_validation_end(self) -> None:
        pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr,
                               )
        optims.append(optimizer)
        return optims