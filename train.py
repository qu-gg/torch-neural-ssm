"""
@file train.py

Holds the general training script for the models, defining a dataset and model to train on
"""
import os
import argparse
import webdataset as wds
import pytorch_lightning

from utils.utils import get_exp_versions, get_model


class Dataset(pytorch_lightning.LightningDataModule):
    def __init__(self, args, batch_size=32, workers=0):
        super(Dataset, self).__init__()
        shard_size = (args.dataset_size // 50) - 1

        bucket = "data/{}/{}/train_tars/".format(args.dataset, args.dataset_ver)
        shards = "{000.." + str(shard_size) + "}.tar"
        self.training_urls = os.path.join(bucket, shards)
        print(self.training_urls)

        bucket = "data/{}/{}/test_tars/".format(args.dataset, args.dataset_ver)
        shards = "{000.." + str(shard_size) + "}.tar"
        self.validation_urls = os.path.join(bucket, shards)

        self.length = args.dataset_size // batch_size
        self.batch_size = batch_size
        self.num_workers = workers

    def make_loader(self, urls, mode="train"):
        shuffle = 1000 if mode == "train" else 0

        dataset = (
            wds.WebDataset(urls, shardshuffle=True)
            .shuffle(shuffle)
            .decode("rgb")
            .to_tuple("npz")
            .batched(self.batch_size, partial=False)
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers
        )

        loader.length = self.length
        loader.unbatched().shuffle(1000).batched(self.batch_size)
        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, "train")

    def val_dataloader(self):
        return self.make_loader(self.validation_urls, "val")


def parse_args():
    """ General arg parsing for non-model parameters """
    parser = argparse.ArgumentParser()

    # Experiment ID
    parser.add_argument('--exptype', type=str, default='pendulum', help='experiment folder name')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')
    parser.add_argument('--model', type=str, default='lstm', help='which model to use for training')

    # Dataset-to-use parameters
    parser.add_argument('--dataset', type=str, default='pendulum', help='dataset name for training')
    parser.add_argument('--dataset_ver', type=str, default='pendulum_50000samples_65steps',
                        help='dataset version for training')
    parser.add_argument('--dataset_size', type=int, default=50000, help='dataset name for training')

    # Learning parameters
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to run over')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='initial learning rate')

    # Tuning parameters
    parser.add_argument('--z0_beta', type=float, default=0.1, help='multiplier for z0 term in loss')
    parser.add_argument('--kl_beta', type=float, default=0.001, help='multiplier for encoder kl terms in loss')

    # Input dimensions
    parser.add_argument('--dim', type=int, default=32, help='dimension of the image data')

    # Latent network dimensions
    parser.add_argument('--latent_dim', type=int, default=16, help='latent dimension size')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in the dynamics func')
    parser.add_argument('--num_hidden', type=int, default=250, help='number of nodes per layer in dynamics func')
    parser.add_argument('--latent_act', type=str, default="leaky_relu", help='type of act func in dynamics func')

    # Convolutional dimensions
    parser.add_argument('--z_amort', type=int, default=5, help='how many X samples to use in z0 inference')
    parser.add_argument('--fix_variance', type=bool, default=False, help='whether to fix variance in z0 encoding')
    parser.add_argument('--num_filt', type=int, default=8, help='number of filters in the CNNs')

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

    # Parse args and manually specify GPU ranks to train on
    arg = parser.parse_args()
    arg.gpus = [0]

    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125)

    # Get version numbers
    global top, exptop
    top, exptop = get_exp_versions(arg.model, arg.exptype)

    # Input generation
    dataset = Dataset(args=arg, batch_size=arg.batch_size)

    # Initialize model
    model = model_type(arg, top, exptop, dataset.length)

    # Initialize pytorch lighthning trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(arg, max_epochs=arg.num_epochs, auto_select_gpus=True, profiler="simple")

    # Start training from scratch or a checkpoint
    if arg.checkpt == 'None':
        trainer.fit(model, dataset)
    else:
        trainer.fit(
            model, dataset,
            ckpt_path="lightning_logs/version_{}/checkpoints/{}".format(
                arg.checkpt, os.listdir("lightning_logs/version_{}/checkpoints/".format(arg.checkpt))[0])
        )
