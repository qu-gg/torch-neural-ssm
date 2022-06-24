"""
@file CommonDynamics.py

A common class that each latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
"""
import os
import torch
import shutil
import numpy as np
import torch.nn as nn
import pytorch_lightning

from utils.metrics import vpt, dst, r2fit
from utils.plotting import show_images
from models.CommonVAE import LatentStateEncoder, EmissionDecoder


class LatentDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args, top, exptop, last_train_idx):
        """
        Generic implementation of a Latent Dynamics Model
        Holds the training and testing boilerplate, as well as experiment tracking
        :param args: passed in user arguments
        :param top: top lightning log version
        :param exptop: top experiment folder version
        """
        super().__init__()
        self.save_hyperparameters(args)

        # Args
        self.args = args
        self.top = top
        self.exptop = exptop
        self.last_train_idx = last_train_idx

        # Encoder + Decoder
        self.encoder = LatentStateEncoder(self.args.z_amort, self.args.num_filt, 1, self.args.latent_dim, self.args.fix_variance)
        self.decoder = EmissionDecoder(self.args.batch_size, self.args.generation_len, self.args.dim, self.args.num_filt, 1, self.args.latent_dim)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def forward(self, x):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        return parent_parser

    def configure_optimizers(self):
        """
        Most standard NSSM models have a joint optimization step under one ELBO, however there is room
        for EM-optimization procedures based on the PGM.

        By default, we assume a joint optim with the Adam Optimizer.
        """
        optim = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optim

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Stack batch and restrict to generation length
        images = torch.stack([b['image'] for b in batch[0]])
        states = torch.stack([b['x'] for b in batch[0]]).squeeze(1)
        images = images[:, :self.args.generation_len]
        states = states[:, :self.args.generation_len]

        # Get predictions
        output = self(images)
        if len(output) == 2:
            preds, embeddings = output[0], output[1]
        else:
            preds, embeddings = output, None

        # Reconstruction loss
        likelihood = self.reconstruction_loss(preds, images).sum([2, 3]).view([-1]).mean()
        self.log("likelihood", likelihood, prog_bar=True)

        # Initial encoder loss
        klz = self.encoder.kl_z_term()
        self.log("klz_loss", self.args.z0_beta * klz, prog_bar=True)

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss()
        self.log("dynamics_loss", dynamics_loss)

        # Build the full loss
        loss = likelihood + (self.args.z0_beta * klz) + dynamics_loss

        # Log various metrics
        self.log("train_vpt", vpt(images, preds.detach()), prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_pixel_mse", likelihood / (self.args.dim ** 2), prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Return outputs as dict
        out = {"loss": loss, "states": states.detach(), "embeddings": embeddings.detach()}
        if batch_idx >= self.last_train_idx - 25:
            out["preds"] = preds.detach()
            out["images"] = images.detach()
        return out

    def training_epoch_end(self, outputs):
        """
        Upon epoch end, save the training file and show reconstructions every 10 epochs
        :param outputs: list of outputs from the training steps, with the last 25 steps having reconstructions
        """
        # Make lightning logs folder if it does not exist
        version_path = os.path.abspath('') + "/lightning_logs/version_{}/".format(self.top)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(version_path + '/images/'):
            os.mkdir(version_path + '/images/')

        # Get a training reconstruction sample and copy log to experiment folder
        if self.current_epoch % 4 == 0:
            # Show side-by-side reconstructions
            show_images(outputs[-1]["images"], outputs[-1]["preds"],
                        version_path + '/images/recon{}train.png'.format(self.current_epoch),
                        num_out=5)

            # Get per-dynamics plots
            self.model_specific_plotting(version_path)

            # Get R^2 over training set
            embeddings, states = [], []
            for out in outputs[::8]:
                embeddings.append(out['embeddings'])
                states.append(out['states'])
            embeddings, states = torch.vstack(embeddings), torch.vstack(states)

            # Get polar coordinates (sin and cos) of the angle for evaluation
            sins = torch.sin(states[:, :, 0])
            coss = torch.cos(states[:, :, 0])
            states = torch.stack((sins, coss, states[:, :, 1]), dim=2)

            # Get r2 score
            r2s = r2fit(embeddings, states, mlp=True)

            # Log each dimension's R2 individually
            for idx, r in enumerate(r2s):
                self.log("train_r2_{}".format(idx), r, prog_bar=True)

            # Copy experiment to relevant folder
            if self.args.exptype is not None and self.args.tune is False:
                if os.path.exists("experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, self.exptop)):
                    shutil.rmtree("experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, self.exptop))
                shutil.copytree("lightning_logs/version_{}/".format(self.top),
                                "experiments/{}/{}/version_{}".format(self.args.model, self.args.exptype, self.exptop))

    def model_specific_plotting(self, version_path):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Stack batch and restrict to generation length
        images = torch.stack([b['image'] for b in batch[0]])
        states = torch.stack([b['x'] for b in batch[0]]).squeeze(1)
        images = images[:, :self.args.generation_len]
        states = states[:, :self.args.generation_len]

        # Get predictions
        output = self(images)
        if len(output) == 2:
            preds, embeddings = output[0], output[1]
        else:
            preds, embeddings = output, None

        # Reconstruction loss
        likelihood = self.reconstruction_loss(preds, images).sum([2, 3]).view([-1]).mean()
        self.log("val_likelihood", likelihood, prog_bar=True, on_epoch=True, on_step=False)

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss()

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Log various metrics
        self.log("val_vpt", vpt(images, preds.detach()), prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_pixel_mse", likelihood / (self.args.dim ** 2), prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Return outputs as dict
        out = {"loss": loss, "states": states.detach(), "embeddings": embeddings.detach()}
        if batch_idx == 0:
            out["preds"] = preds.detach()
            out["images"] = images.detach()
        return out

    def validation_epoch_end(self, outputs):
        """
        Upon epoch end, save the training file and show reconstructions every 5 epochs
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Make lightning logs folder if it does not exist
        version_path = os.path.abspath('') + "/lightning_logs/version_{}/".format(self.top)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(version_path + '/images/'):
            os.mkdir(version_path + '/images/')

        # Every 4 epochs get a training reconstruction sample and copy log to experiment folder
        if self.current_epoch % 5 == 0 and self.current_epoch != 0:
            # Show side-by-side reconstructions
            show_images(outputs[0]["images"], outputs[0]["preds"],
                        version_path + '/images/recon{}val.png'.format(self.current_epoch),
                        num_out=5)

            # Get R^2 over validation set
            embeddings, states = [], []
            for out in outputs[::8]:
                embeddings.append(out['embeddings'])
                states.append(out['states'])
            embeddings, states = torch.vstack(embeddings), torch.vstack(states)

            # Get polar coordinates (sin and cos) of the angle for evaluation
            sins = torch.sin(states[:, :, 0])
            coss = torch.cos(states[:, :, 0])
            states = torch.stack((sins, coss, states[:, :, 1]), dim=2)

            # Get r2 score
            r2s = r2fit(embeddings, states, mlp=True)

            # Log each dimension's R2 individually
            for idx, r in enumerate(r2s):
                self.log("val_r2_{}".format(idx), r, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step. For every batch, get the predictions and per_pixel MSE averages over time
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        images = torch.stack([b['image'] for b in batch[0]])

        # Get predictions and C embeddings
        output = self(images)
        preds, embeddings = output if len(output) > 1 else (output, None)

        # Reconstruction loss
        loss = self.reconstruction_loss(preds, images).sum([2, 3]).view([-1]).mean()

        # Per pixel MSE loss
        pixel_mse = self.bce(preds, images)
        pixel_mse = pixel_mse.sum([2]).view([preds.shape[0], preds.shape[1], -1])
        pixel_mse = pixel_mse.mean([2])
        return {"loss": loss, "pixel_mse": pixel_mse.detach(), "preds": preds.detach(), "images": images.detach()}

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        pixel_mse, preds, images = [], [], []

        # Compile and save reconstructions vs GT to files
        for output in outputs:
            pixel_mse.append(output["pixel_mse"])
            preds.append(output["preds"])
            images.append(output["images"])

        # Stack tensors
        pixel_mse = torch.vstack(pixel_mse)
        preds = torch.vstack(preds).detach().cpu().numpy()
        images = torch.vstack(images).detach().cpu().numpy()

        # Save files
        ckpt_path = "experiments/{}/{}/version_{}/".format(self.args.model, self.args.exptype, self.exptop)
        np.save("{}/pixel_mse.npy".format(ckpt_path), pixel_mse)
        np.save("{}/recons.npy".format(ckpt_path), preds)
        np.save("{}/images.npy".format(ckpt_path), images)
