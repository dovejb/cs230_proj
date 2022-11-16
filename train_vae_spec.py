from vae_spec import SpecVAE
from constants import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets import *
from lightning_module_spec import VocalSeparatorSpec

if __name__ == '__main__':
    model = SpecVAE(WAV_SHAPE[0], latent_dim=128, beta=0.00001)
    from torchinfo import summary
    summary(model, input_size=(32,1,WAV_SHAPE[0]))
    tb_logger = TensorBoardLogger(save_dir="./log",
                                  name="spec")
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
                      #max_epochs=EPOCHS,
                      log_every_n_steps=5,
                      gradient_clip_val=0.5,
                      auto_scale_batch_size="binsearch",
                      resume_from_checkpoint='./log/spec/version_2/checkpoints/last.ckpt',
                      check_val_every_n_epoch=5,
                      )
    data = MyDataset()
    module = VocalSeparatorSpec(model, LEARNING_RATE)

    trainer.fit(module, data)