"""
@file main.py

Main entrypoint for the training and testing environments. Takes in a configuration file
of arguments and either trains a model or tests a given model and checkpoint.
"""
import torch
import hydra
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader import SSMDataModule
from utils.utils import get_model, flatten_cfg
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Flatten the cfg down a level
    cfg.expname = cfg.expname
    cfg = flatten_cfg(cfg)

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(cfg.seed, workers=True)

    # Enable fp16 training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # Limit number of CPU workers
    torch.set_num_threads(8)

    # Building the PL-DataModule for all splits
    datamodule = SSMDataModule(cfg=cfg)

    # Initialize model type and initialize
    model = get_model(cfg.model_type)(cfg)

    # Tensorboard Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.expname}/", name=f"{cfg.model_type}")

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_reconstruction_mse',
                                          filename='step{step:02d}-val_reconstruction_mse{val_reconstruction_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_reconstruction_mse", min_delta=0.000001, patience=15, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer(
        callbacks=[
            # early_stop_callback,
            lr_monitor,
            checkpoint_callback
        ],
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        deterministic=True,
        max_steps=cfg.num_steps * cfg.batch_size,
        max_epochs=1,
        gradient_clip_val=5.0,
        val_check_interval=cfg.metric_interval,
        num_sanity_val_steps=0,
        logger=tb_logger
    )

    # Training from scratch
    if cfg.train is True and cfg.resume is False:
        trainer.fit(model, datamodule)

    # Training from the last epoch
    elif cfg.train is True and cfg.resume is True:
        ckpt_path = tb_logger.log_dir + "/checkpoints/" + cfg.checkpt if cfg.checkpt != "" \
            else f"{tb_logger.log_dir}/checkpoints/last.ckpt"

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Testing the model on each split
    ckpt_path = tb_logger.log_dir + "/checkpoints/" + cfg.checkpt if cfg.checkpt != "" \
            else f"{tb_logger.log_dir}/checkpoints/last.ckpt"

    model.setting = 'train'
    trainer.test(model, datamodule.evaluate_train_dataloader(), ckpt_path=ckpt_path)

    model.setting = 'val'
    trainer.test(model, datamodule.val_dataloader(), ckpt_path=ckpt_path)

    model.setting = 'test'
    trainer.test(model, datamodule.test_dataloader(), ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
