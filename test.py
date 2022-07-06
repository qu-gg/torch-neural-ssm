"""
@file test.py
@author Ryan Missel

Handles loading in the checkpoint of a model and testing it on a given dataset for metrics, plots, and reconstructions.
"""
import json
import os
import argparse
import pytorch_lightning

from utils.dataloader import Dataset
from utils.utils import get_exp_versions, get_model


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID
    parser.add_argument('--exptype', type=str, default='pendulum_3latent', help='experiment folder name')
    parser.add_argument('--ckpt_path', type=str, default='experiments_tuned/nODEtuned/')
    parser.add_argument('--model', type=str, default='node_si', help='which model to use for training')

    # Dataset-to-use parameters
    parser.add_argument('--dataset', type=str, default='pendulum', help='dataset name for training')
    parser.add_argument('--dataset_ver', type=str, default='pendulum_250samples_1000steps',
                        help='dataset version for training')
    parser.add_argument('--dataset_size', type=int, default=250, help='dataset name for training')

    # Input dimensions
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')

    # Convolutional dimensions
    parser.add_argument('--z_amort', type=int, default=5, help='how many X samples to use in z0 inference')

    # Timesteps of generation
    parser.add_argument('--generation_len', type=int, default=65, help='total length to generate')
    return parser


if __name__ == '__main__':
    # Parse cmd line args
    parser = parse_args()
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    # Get the model type from args and add its specific arguments
    model_type = get_model(parser.parse_args().model)
    parser = model_type.add_model_specific_args(parser)

    # Parse args
    arg = parser.parse_args()

    # Set tuning mode to True and manually specify GPU ranks to train on
    arg.tune = False
    arg.gpus = [0]

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Build the checkpoint path
    ckpt_path = f"{arg.ckpt_path}/checkpoints/{os.listdir(f'{arg.ckpt_path}/checkpoints/')[0]}"
    print(ckpt_path)

    import torch
    ckpt = torch.load(ckpt_path)

    # Load in hyperparameter JSON and add to argparse NameSpace
    with open(f"{arg.ckpt_path}/params.json", 'r') as f:
        params = json.load(f)
    argdict = vars(arg)
    argdict.update(params)
    arg = argparse.Namespace(**argdict)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Input generation
    dataset = Dataset(args=arg, batch_size=arg.batch_size)

    # Initialize model
    model = model_type(arg, top, exptop, dataset.length)

    # Initialize pytorch lighthning trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, max_epochs=1, auto_select_gpus=True)

    # Test model on the given set and checkpoint
    trainer.test(model, dataset, ckpt_path=ckpt_path)
