"""
@file train.py

Holds the general training script for the models, defining a dataset and model to train on
"""
import os
import argparse
import pytorch_lightning

from utils.dataloader import Dataset
from utils.utils import get_exp_versions, get_model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID and Checkpoint to Load
    parser.add_argument('--exptype', type=str, default='pendulum', help='experiment folder name')
    parser.add_argument('--ckpt_path', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--dev', type=int, default=0, help='which gpu device to use')

    # Defining which model and model version to use
    parser.add_argument('--model', type=str, default='node', help='choice of latent dynamics function')
    parser.add_argument('--system_identification', type=bool, default=True,
                        help='whether to use (True) system identification or (False) state estimation model versions'
                             'note that some baselines ignore this parameter and are fixed')

    # Dataset-to-use parameters
    parser.add_argument('--dataset', type=str, default='pendulum', help='dataset folder name')
    parser.add_argument('--dataset_ver', type=str, default='pendulum_10000samples_200steps', help='dataset version')
    parser.add_argument('--dataset_percent', type=int, default=1.0, help='percent of dataset to use')
    parser.add_argument('--batches_to_save', type=int, default=25, help='how many batches to output per epoch')

    # Learning parameters
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='initial learning rate')

    # Tuning parameters
    parser.add_argument('--z0_beta', type=float, default=0.01, help='multiplier for z0 term in loss')
    parser.add_argument('--kl_beta', type=float, default=0.001, help='multiplier for dynamic specific kl terms in loss')

    # Input dimensions
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')

    # Network dimensions
    parser.add_argument('--latent_dim', type=int, default=24, help='latent dimension size')
    parser.add_argument('--latent_act', type=str, default="leaky_relu", help='type of act func in dynamics func')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in the dynamics func')
    parser.add_argument('--num_hidden', type=int, default=256, help='number of nodes per layer in dynamics func')
    parser.add_argument('--num_filt', type=int, default=16, help='number of filters in the CNNs')

    # Z0 inference parameters
    parser.add_argument('--z_amort', type=int, default=5, help='how many true frames to use in z0 inference')
    parser.add_argument('--fix_variance', type=bool, default=False, help='whether to fix variance in z0 encoding')

    # Timesteps to generate out
    parser.add_argument('--generation_len', type=int, default=10, help='total length to generate (including z_amort)')
    return parser


if __name__ == '__main__':
    # Parse cmd line args
    parser = parse_args()
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    # Get the model type from args and add its specific arguments
    model_type = get_model(parser.parse_args().model, parser.parse_args().system_identification)
    parser = model_type.add_model_specific_args(parser)

    # Parse args
    arg = parser.parse_args()

    # Set tuning mode to True and manually specify GPU ranks to train on
    arg.tune = False
    arg.gpus = [arg.dev]

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Input generation
    dataset = Dataset(args=arg, batch_size=arg.batch_size)

    # Initialize model
    model = model_type(arg, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_pixel_mse',
                                          filename='epoch{epoch:02d}-val_pixel_mse{val_pixel_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_pixel_mse", min_delta=0.0005, patience=10, mode="min")

    # Initialize trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, callbacks=[early_stop_callback, checkpoint_callback],
                                                           max_epochs=arg.num_epochs, check_val_every_n_epoch=4,
                                                           auto_select_gpus=True, reload_dataloaders_every_epoch=True)
    # Start training from scratch or a checkpoint
    if arg.ckpt_path == 'None':
        trainer.fit(model, dataset)
    else:
        trainer.fit(
            model, dataset,
            ckpt_path=f"{arg.ckpt_path}/checkpoints/{os.listdir(f'{arg.ckpt_path}/checkpoints/')[-1]}"
        )
