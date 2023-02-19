"""
@file test.py
@author Ryan Missel

Handles loading in the checkpoint of a model and testing it on a given dataset for metrics, plots, and reconstructions.
"""
import os
import json
import torch
import argparse
import pytorch_lightning

from distutils.util import strtobool
from utils.dataloader import Dataset
from utils.utils import get_exp_versions, get_model, StoreDictKeyPair


def parse_args():
    """ General arg parsing """
    parser = argparse.ArgumentParser()

    # Experiment ID and Checkpoint to Load
    parser.add_argument('--exptype', type=str, default='testing', help='experiment folder name')
    parser.add_argument('--ckpt_path', type=str, default='lightning_logs/version_3',
                        help='path to the checkpoints folder')
    parser.add_argument('--checkpt', type=str, default='None',
                        help='name a specific checkpoint, will use the last one if none given')
    parser.add_argument('--dev', type=int, default=0, help='which gpu device to use')

    # Whether to save output files (useful for visualizations, not great for storage space)
    parser.add_argument('--save_files', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether to save dataset files in testing experiments')

    # Metrics to evaluate on
    parser.add_argument('--metrics', type=list,
                        default=['vpt', 'dst', 'vpd', 'reconstruction_mse', 'extrapolation_mse'],
                        help='which metrics to use')

    # Defining which model and model version to use
    parser.add_argument('--model', type=str, default='node', help='choice of latent dynamics function')
    parser.add_argument('--system_identification', type=bool, default=True,
                        help='whether to use (True) system identification or (False) state estimation model versions'
                             'note that some baselines ignore this parameter and are fixed')
    parser.add_argument('--stochastic', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether the dynamics parameters are stochastic')

    # ODE Integration parameters
    parser.add_argument('--integrator', type=str, default='symplectic', help='which ODE integrator to use')
    parser.add_argument('--integrator_params', dest="integrator_params",
                        action=StoreDictKeyPair, default={'step_size': 0.5},
                        help='ODE integrator options, set as --integrator_params key1=value1,key2=value2,...')

    # Dataset-to-use parameters
    parser.add_argument('--dataset', type=str, default='pendulum', help='dataset folder name')
    parser.add_argument('--dataset_ver', type=str, default='pendulum_12500samples_200steps_dt01', help='dataset version')
    parser.add_argument('--dataset_percent', type=float, default=1.0, help='how much of the dataset to use')

    # Input dimensions
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for testing')

    # Convolutional dimensions
    parser.add_argument('--z_amort', type=int, default=5, help='how many X samples to use in z0 inference')

    # Timesteps of generation
    parser.add_argument('--generation_len', type=int, default=90, help='total length to generate')
    parser.add_argument('--training_len', type=int, default=15, help='total length to generate')
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
    pytorch_lightning.seed_everything(125125125, workers=True)

    # Build the checkpoint path and load it
    if arg.checkpt != "None":
        ckpt_path = arg.ckpt_path + "/checkpoints/" + arg.checkpt
    else:
        ckpt_path = f"{arg.ckpt_path}/checkpoints/{os.listdir(f'{arg.ckpt_path}/checkpoints/')[-1]}"
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{arg.gpus[0]}")
    print(ckpt_path)

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
    model = model_type(arg, top, exptop)

    # Initialize pytorch lighthning trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, max_epochs=1, deterministic=True, auto_select_gpus=True)

    # Test model on the given set and checkpoint
    trainer.test(model, dataset, ckpt_path=ckpt_path)
