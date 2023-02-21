"""
@file main.py

Main entrypoint for the training and testing environments. Takes in a configuration file
of arguments and either trains a model or tests a given model and checkpoint.
"""
import os
import argparse
import pytorch_lightning

from utils.dataloader import Dataset
from utils.utils import parse_args, get_exp_versions, strtobool
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/CONFIG_EXAMPLE.json',
                        help='path and name of the configuration .json to load')
    parser.add_argument('--train', type=strtobool, default=True,
                        help='whether to train or test the given model')
    parser.add_argument('--resume', type=strtobool, default=False,
                        help='whether to continue training from the checkpoint in the config')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Input generation
    dataset = Dataset(args=args, batch_size=args.batch_size)

    # Initialize model
    model = model_type(args, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_reconstruction_mse',
                                          filename='epoch{epoch:02d}-val_reconstruction_mse{val_reconstruction_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_mse_recon", min_delta=0.0005, patience=10, mode="min")
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
        max_epochs=args.num_epochs,
        gradient_clip_val=5.0,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        auto_select_gpus=True
    )

    if args.train is True and args.resume is False:
        trainer.fit(model, dataset)
    else:
        # Get the checkpoint - either a given one or the last.ckpt in the folder
        if args.checkpt != "None":
            ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt
        else:
            ckpt_path = f"{args.ckpt_path}/checkpoints/{os.listdir(f'{args.ckpt_path}/checkpoints/')[-1]}"

        # If it is training, then resume from the given checkpoint
        if args.train is True:
            trainer.fit(
                model, dataset,
                ckpt_path=f"{args.ckpt_path}/checkpoints/{os.listdir(f'{args.ckpt_path}/checkpoints/')[-1]}"
            )

        # Otherwise test the model
        else:
            trainer.test(model, dataset, ckpt_path=ckpt_path)
