"""
@file CommonDynamics.py

A common class that each latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
"""
import os
import json
import torch
import shutil
import numpy as np
import torch.nn as nn
import pytorch_lightning

from utils.metrics import vpt, dst, r2fit, vpd
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

    def model_specific_loss(self, x, x_rec, zts, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

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

    def on_train_start(self):
        # Get total number of parameters for the model and save
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Save hyperparameters to the log
        params = {
            'latent_dim': self.args.latent_dim,
            'num_layers': self.args.num_layers,
            'num_hidden': self.args.num_hidden,
            'num_filt': self.args.num_filt,
            'latent_act': self.args.latent_act,
            'z_amort': self.args.z_amort,
            'fix_variance': self.args.fix_variance,
            'number_params': pytorch_total_params
        }
        with open(f"lightning_logs/version_{self.top}/params.json", 'w') as f:
            json.dump(params, f)

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
        labels = torch.stack([b['class_id'] for b in batch[0]])
        images = images[:, :self.args.generation_len]
        states = states[:, :self.args.generation_len]

        # Get predictions
        output = self(images)
        if len(output) == 2:
            preds, embeddings = output[0], output[1]
        else:
            preds, embeddings = output, None

        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds[:, 1:], images[:, 1:]).sum([1, 2, 3]).view([-1]).mean()
        self.log("likelihood", likelihood, prog_bar=True)

        likelihood_init = self.reconstruction_loss(preds[:, 0], images[:, 0]).sum([1, 2]).view([-1]).mean()
        self.log("likelihood_init", likelihood_init, prog_bar=True)

        # Initial encoder loss
        klz = self.encoder.kl_z_term()
        self.log("klz_loss", self.args.z0_beta * klz, prog_bar=True)

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(images, preds, embeddings)
        self.log("dynamics_loss", dynamics_loss)

        # Build the full loss
        loss = likelihood + (10 * likelihood_init) + (self.args.z0_beta * klz) + dynamics_loss

        # Log various metrics
        self.log("train_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_pixel_mse", self.reconstruction_loss(preds[:, 1:], images[:, 1:]).mean([1, 2, 3]), prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Return outputs as dict
        out = {"loss": loss, "states": states.detach(), "embeddings": embeddings.detach(), "labels": labels.detach()}
        if batch_idx >= self.last_train_idx - 25:
            out["preds"] = preds.detach()
            out["images"] = images.detach()

            # For code vector based models (i.e. the proposed models) also add their local codes
            if hasattr(self.dynamics_func, 'embeddings'):
                out['code_vectors'] = self.dynamics_func.embeddings.detach()
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

        # Every 4 epochs, get a reconstruction example, model-specific plots, and copy over to the experiments folder
        if self.current_epoch % 4 == 0:
            # Show side-by-side reconstructions
            show_images(outputs[-1]["images"], outputs[-1]["preds"],
                        version_path + f'/images/recon{self.current_epoch}train.png', num_out=5)

            # Get per-dynamics plots
            self.model_specific_plotting(version_path, outputs)

            # Copy experiment to relevant folder
            if self.args.exptype is not None:
                if os.path.exists("experiments/{}/{}/version_{}".format(self.args.exptype, self.args.model, self.exptop)):
                    shutil.rmtree("experiments/{}/{}/version_{}".format(self.args.exptype, self.args.model, self.exptop))
                shutil.copytree("lightning_logs/version_{}/".format(self.top),
                                "experiments/{}/{}/version_{}".format(self.args.exptype, self.args.model, self.exptop))

        # Every 8 epochs, get a R^2 fit metric on the latent states to gt states
        if self.current_epoch % 8 == 0:
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
                self.log("train_r2_{}".format(idx), r, prog_bar=False)

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

        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds[:, 1:], images[:, 1:]).sum([1, 2, 3]).view([-1]).mean()
        self.log("val_likelihood", likelihood, prog_bar=True)

        likelihood_init = self.reconstruction_loss(preds[:, 0], images[:, 0]).sum([1, 2]).view([-1]).mean()
        self.log("val_likelihood_init", likelihood_init, prog_bar=True)

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(images, preds, embeddings, train=False)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Log various metrics
        self.log("val_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_pixel_mse", self.reconstruction_loss(preds[:, 1:], images[:, 1:]).mean([1, 2, 3]), prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

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

        if self.current_epoch % 8 == 0:
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
                self.log("val_r2_{}".format(idx), r, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step. For every batch, get the predictions and per_pixel MSE averages over time
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

        return {"states": states.detach(), "embeddings": embeddings.detach(),
                "preds": preds.detach(), "images": images.detach()}

    def test_epoch_end(self, outputs):
        """
        For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Stack all outputs
        preds, images, states, embeddings = [], [], [], []
        for output in outputs:
            preds.append(output["preds"])
            images.append(output["images"])
            states.append(output["states"])
            embeddings.append(output["embeddings"])

        preds = torch.vstack(preds)
        images = torch.vstack(images)
        states = torch.vstack(states)
        embeddings = torch.vstack(embeddings)

        # Print statistics over the full set
        pixel_mse = self.reconstruction_loss(preds[:, 1:], images[:, 1:]).detach().cpu().numpy()
        vpt_mean, vpt_std = vpt(images, preds.detach())
        dsts = dst(images, preds.detach())[0]
        vpds = vpd(images, preds.detach())
        print("")
        print(f"=> Pixel MSE: {np.mean(pixel_mse):4.5f}+-{np.std(pixel_mse):4.5f}")
        print(f"=> VPT:       {vpt_mean:4.5f}+-{vpt_std:4.5f}")
        print(f"=> DST:       {np.mean(dsts):4.5f}+-{np.std(dsts):4.5f}")
        print(f"=> VPD:       {np.mean(vpds):4.5f}+-{np.std(vpds):4.5f}")

        metrics = {
            "mse_mean": float(np.mean(pixel_mse)),
            "mse_std": float(np.std(pixel_mse)),
            "vpt_mean": float(vpt_mean),
            "vpt_std": float(vpt_std),
            "dst_mean": float(np.mean(dsts)),
            "dst_std": float(np.std(dsts)),
            "vpd_mean": float(np.mean(vpds)),
            "vpd_std": float(np.std(vpds))
        }

        # Get polar coordinates (sin and cos) of the angle for evaluation
        sins = torch.sin(states[:, :, 0])
        coss = torch.cos(states[:, :, 0])
        states = torch.stack((sins, coss, states[:, :, 1]), dim=2)

        # Get r2 score
        r2s = r2fit(embeddings, states, mlp=True)

        # Log each dimension's R2 individually
        for idx, r in enumerate(r2s):
            metrics[f"r2_{idx}"] = r

        # Save files
        np.save(f"{self.args.checkpt}/pixel_mse.npy", pixel_mse)
        np.save(f"{self.args.checkpt}/test_recons.npy", preds.detach().cpu().numpy())
        np.save(f"{self.args.checkpt}/test_images.npy", images.detach().cpu().numpy())

        # Save some examples
        show_images(images[:10], preds[:10], f"{self.args.checkpt}/test_{self.args.split}_examples.png", num_out=5)

        # Save metrics to JSON in checkpoint folder
        with open(f"{self.args.checkpt}/test_metrics.json", 'w') as f:
            json.dump(metrics, f)