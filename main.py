"""
@file main.py

Main entrypoint for the training and testing environments. Takes in a configuration file
of arguments and either trains a model or tests a given model and checkpoint.
"""
import argparse
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from utils.dataloader import SSMDataModule
from utils.utils import parse_args, strtobool, find_best_step
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/grav16.json', help='config .json to load')
    parser.add_argument('--train', type=strtobool, default=True, help='whether to train or test the given model')
    parser.add_argument('--resume', type=strtobool, default=False, help='whether to continue training from ckpt')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Building the PL-DataModule for all splits
    datamodule = SSMDataModule(args=args)

    # Initialize model
    model = model_type(args)

    # Tensorboard Logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{args.exptype}/{args.model}/")

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_reconstruction_mse',
                                          filename='step{step:02d}-val_reconstruction_mse{val_reconstruction_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_reconstruction_mse", min_delta=0.000001, patience=15, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(
        args,
        callbacks=[
            # early_stop_callback,
            lr_monitor,
            checkpoint_callback
        ],
        deterministic=True,
        max_steps=args.num_steps * args.batch_size,
        max_epochs=1,
        gradient_clip_val=5.0,
        val_check_interval=250,
        num_sanity_val_steps=0,
        auto_select_gpus=True,
        logger=tb_logger
    )

    # Training from scratch
    if args.train is True and args.resume is False:
        trainer.fit(model, datamodule)

    # Training from the last epoch
    elif args.train is True and args.resume is True:
        ckpt_path = tb_logger.log_dir + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{tb_logger.log_dir}/checkpoints/last.ckpt"

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # Testing the model on each split
    ckpt_path = tb_logger.log_dir + "/checkpoints/" + args.checkpt if args.checkpt not in ["", "None"] \
        else f"{tb_logger.log_dir}/checkpoints/{find_best_step(tb_logger.log_dir)[0]}"

    args.setting = 'train'
    trainer.test(model, datamodule.evaluate_train_dataloader(), ckpt_path=ckpt_path)

    args.setting = 'val'
    trainer.test(model, datamodule.val_dataloader(), ckpt_path=ckpt_path)

    args.setting = 'test'
    trainer.test(model, datamodule.test_dataloader(), ckpt_path=ckpt_path)
