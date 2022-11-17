import torch
from vae_spec import SpecVAE
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets import *
from lightning_module_spec import VocalSeparatorSpec
from convtasnet.conv_tasnet import TasNet

NAME="tasnet"

if __name__ == '__main__':
    #model = SpecVAE(WAV_SHAPE[0], latent_dim=128, beta=0)#1e-6)
    model = TasNet(num_spk=1)
    from torchinfo import summary
    summary(model, input_size=(32,1,WAV_SHAPE[0]))
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
                      max_epochs=300,
                      log_every_n_steps=5,
                      gradient_clip_val=0.5,
                      auto_scale_batch_size="binsearch",
                      #resume_from_checkpoint=f"./log/{NAME}/version_2/checkpoints/last.ckpt",
                      check_val_every_n_epoch=3,
                      )
    data = MyDataset()
    module = VocalSeparatorSpec(model, 1e-2)

    trainer.fit(module, data)
    torch.save(module.model, f"./{NAME}.model")