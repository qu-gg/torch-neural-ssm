"""
@file tune.py
@author Ryan Missel

Handles performing a hyperparameter search using RayTune for the given models.
Define a range of discrete or sampled parameters that are randomly sampled and ran for the best validation loss.

Can handle splitting automatically over available GPUs using PyTorch-Lightning
"""
import os
import argparse
from distutils.util import strtobool

import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.schedulers import ASHAScheduler

from utils.dataloader import Dataset
from utils.utils import get_exp_versions, get_model, StoreDictKeyPair

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID and Checkpoint to Load
    parser.add_argument('--exptype', type=str, default='testing', help='experiment folder name')
    parser.add_argument('--ckpt_path', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--dev', type=int, default=0, help='which gpu device to use')

    # Defining which model and model version to use
    parser.add_argument('--model', type=str, default='node', help='choice of latent dynamics function')
    parser.add_argument('--system_identification', type=bool, default=True,
                        help='whether to use (True) system identification or (False) state estimation model versions'
                             'note that some baselines ignore this parameter and are fixed')
    parser.add_argument('--stochastic', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether the dynamics parameters are stochastic')

    # ODE Integration parameters
    parser.add_argument('--integrator', type=str, default='rk4', help='which ODE integrator to use')
    parser.add_argument('--integrator_params', dest="integrator_params",
                        action=StoreDictKeyPair, default={'step_size': 0.5},
                        help='ODE integrator options, set as --integrator_params key1=value1,key2=value2,...')

    # Dataset-to-use parameters
    parser.add_argument('--dataset', type=str, default='pendulum', help='dataset folder name')
    parser.add_argument('--dataset_ver', type=str, default='pendulum_12500samples_200steps_dt01',
                        help='dataset version')
    parser.add_argument('--dataset_percent', type=int, default=1.0, help='percent of dataset to use')
    parser.add_argument('--batches_to_save', type=int, default=25, help='how many batches to output per epoch')

    # Learning parameters
    parser.add_argument('--num_epochs', type=int, default=249, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')

    # Tuning parameters
    parser.add_argument('--z0_beta', type=float, default=0.01, help='multiplier for z0 term in loss')
    parser.add_argument('--kl_beta', type=float, default=0.001, help='multiplier for dynamic specific kl terms in loss')

    # Input dimensions
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')

    # Network dimensions
    parser.add_argument('--latent_dim', type=int, default=32, help='latent dimension size')
    parser.add_argument('--latent_act', type=str, default="swish", help='type of act func in dynamics func')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in the dynamics func')
    parser.add_argument('--num_hidden', type=int, default=256, help='number of nodes per layer in dynamics func')
    parser.add_argument('--num_filt', type=int, default=16, help='number of filters in the CNNs')

    # Z0 inference parameters
    parser.add_argument('--z_amort', type=int, default=5, help='how many true frames to use in z0 inference')

    # Timesteps to generate out
    parser.add_argument('--generation_len', type=int, default=15, help='total length to generate (including z_amort)')
    parser.add_argument('--generation_varying', type=lambda x: bool(strtobool(x)),
                        default=True, help='whether to vary the generation_len/batch')
    parser.add_argument('--generation_validation_len', type=int, default=30,
                        help='total length to generate for validation')
    return parser


def train_model(configs, args, num_epochs):
    # Define metrics of interest
    metrics = {"val_loss": "val_likelihood", "val_vpt": "val_vpt", "val_mse": "val_pixel_mse", "val_dst": "val_dst"}

    # Given this config, set the parser arguments
    # Gross solution but easier to adjust just the tuning side to the general system architecture
    args.batch_size = configs['batch_size']
    args.latent_dim = configs['latent_dim']
    args.num_hidden = configs['num_hidden']
    args.num_layers = configs['num_layers']
    args.num_filt = configs['num_filt']
    args.latent_act = configs['latent_act']

    args.fix_variance = configs['fix_variance']
    args.learning_rate = configs['learning_rate']
    args.batch_size = configs['batch_size']

    # Input generation
    dataset = Dataset(args=args, batch_size=args.batch_size)

    # Initialize model
    model = model_type(args, 1, 1, dataset.length)

    # Initialize directories
    if not os.path.exists(tune.get_trial_dir() + "lightning_logs/"):
        os.mkdir(tune.get_trial_dir() + "lightning_logs/")

    if not os.path.exists(tune.get_trial_dir() + "lightning_logs/version_1/"):
        os.mkdir(tune.get_trial_dir() + "lightning_logs/version_1/")

    # Initialize pytorch lightning trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(args,
                                                           max_epochs=num_epochs,
                                                           auto_select_gpus=True,
                                                           enable_progress_bar=False,
                                                           num_sanity_val_steps=0,
                                                           logger=TensorBoardLogger(
                                                               save_dir=tune.get_trial_dir(), name="", version="."),
                                                           callbacks=[TuneReportCallback(metrics, on="validation_end")]
                                                           )

    # Call trainer fit for this trial
    trainer.fit(model, dataset)
    return


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
    arg.tune = True
    arg.gpus = [0]

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Define how many models to sample and epochs to run
    num_samples = 75
    num_epochs = 3

    # Hyperparameter choices
    config = {
        "latent_dim": tune.choice([3, 6, 12, 16, 24]),
        "num_hidden": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([2, 3, 4]),
        "num_filt": tune.choice([4, 8, 16]),
        "latent_act": tune.choice(["leaky_relu", "tanh", "swish"]),
        "fix_variance": tune.choice([False, True]),

        "learning_rate": tune.loguniform(5e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
    }

    # Hyperparam scheduler
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    # Define tune object
    trainable = tune.with_parameters(
        train_model,
        args=arg,
        num_epochs=num_epochs
    )

    # Run the analysis
    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_node_si",
        local_dir=os.path.abspath("./")
    )

    # Print out the best configuration
    print(analysis.best_config)
