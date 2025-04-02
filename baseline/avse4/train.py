import logging
import time

from utils import seed_everything
seed_everything(1143)
import torch
torch.set_float32_matmul_precision('medium')
from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import AVSE4BaselineModule
from dataset import AVSE4DataModule

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch",
                                          filename="model-{epoch:02d}-{val_loss:.3f}", save_top_k=2, save_last=True)
    callbacks = [checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=6)]
    datamodule = AVSE4DataModule(data_root=cfg.data.root, batch_size=cfg.data.batch_size,
                                 audio_norm=cfg.data.audio_norm, rgb=cfg.data.rgb,
                                 num_channels=cfg.data.num_channels)
    model = AVSE4BaselineModule(num_channels=cfg.data.num_channels)
    
    trainer = Trainer(default_root_dir=cfg.trainer.log_dir,
                      callbacks=callbacks, deterministic=cfg.trainer.deterministic,
                      log_every_n_steps=cfg.trainer.log_every_n_steps,
                      fast_dev_run=cfg.trainer.fast_dev_run, devices=cfg.trainer.gpus,
                      accelerator=cfg.trainer.accelerator,
                      precision=cfg.trainer.precision, strategy=cfg.trainer.strategy,
                      max_epochs=cfg.trainer.max_epochs,
                      accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                      detect_anomaly=cfg.trainer.detect_anomaly,
                      limit_train_batches=cfg.trainer.limit_train_batches,
                      limit_val_batches=cfg.trainer.limit_val_batches,
                      num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
                      gradient_clip_val=cfg.trainer.gradient_clip_val,
                      profiler=cfg.trainer.profiler
                      )
    start = time.time()
    trainer.fit(model, datamodule, ckpt_path=cfg.trainer.ckpt_path)
    log.info(f"Time taken {time.time() - start} sec")


if __name__ == '__main__':
    main()
