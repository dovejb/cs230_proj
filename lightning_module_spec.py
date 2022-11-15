import pytorch_lightning as pl
from torch import nn, Tensor
from torch import optim

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
        results = self.forward(x)
        train_loss = self.model.loss_function(y,
                                            *results, 
                                            optimizer_idx=optimizer_idx, 
                                            batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        results = self.forward(x)
        val_loss = self.model.loss_function(y, 
                                            *results, 
                                            optimizer_idx=optimizer_idx, 
                                            batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

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