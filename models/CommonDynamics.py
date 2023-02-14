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

from utils.plotting import show_images, get_embedding_trajectories
from utils.metrics import vpt, dst, r2fit, vpd, normalized_pixel_mse
from utils.utils import determine_annealing_factor, CosineAnnealingWarmRestartsWithDecayAndLinearWarmup
from models.CommonVAE import LatentStateEncoder, EmissionDecoder


class LatentDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args, top, exptop):
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

        # Encoder + Decoder
        self.encoder = LatentStateEncoder(self.args.z_amort, self.args.num_filt, 1, self.args.latent_dim, self.args.stochastic)
        self.decoder = EmissionDecoder(self.args.batch_size, self.args.generation_len, self.args.dim, self.args.num_filt, 1, self.args.latent_dim)

        # Recurrent dynamics function
        self.dynamics_func = None
        self.dynamics_out = None

        # Number of steps for training
        self.n_updates = 0

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def forward(self, x, generation_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward function: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, x_rec, zts, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

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

        By default, we assume a joint optim with the Adam Optimizer. We additionally include LR Warmup and
        CosineAnnealing with decay for standard learning rate care during training.

        For CosineAnnealing, we set the LR bounds to be [LR * 1e-2, LR]
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(optim, T_0=5000, T_mult=1,
                                                                        eta_min=self.args.learning_rate * 1e-2,
                                                                        warmup_steps=600, decay=0.90)

        # Explicit dictionary to state how often to ping the scheduler
        scheduler = {
            'scheduler': scheduler,
            'frequency': 1,
            'interval': 'step'
        }
        return [optim], [scheduler]

    def on_train_start(self):
        """
        Before a training session starts, we set some model variables and save a JSON configuration of the
        used hyper-parameters to allow for easy load-in at test-time.
        """
        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

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
            'stochastic': self.args.stochastic,
            'system_identification': self.args.system_identification,
            'learning_rate': self.args.learning_rate,
            'number_params': pytorch_total_params,
        }
        with open(f"{self.version_path}/params.json", 'w') as f:
            json.dump(params, f)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/images/"):
            os.mkdir(f"{self.version_path}/images/")

    def on_validation_start(self):
        """
        Extra check to make sure the version path variable is set when restarting from a given checkpoint.
        Sometimes it can start on a reconstruction-based epoch and cause errors without this check.
        """
        # Get local version path from absolute directory
        self.version_path = f"{os.path.abspath('')}/lightning_logs/version_{self.top}/"

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.version_path}/images/"):
            os.mkdir(f"{self.version_path}/images/")

    def get_step_outputs(self, batch, generation_len):
        """
        Handles the process of pre-processing and subsequence sampling a batch,
        as well as getting the outputs from the models regardless of step
        :param batch: list of dictionary objects representing a single image
        :param generation_len: how far out to generate for, dependent on the step (train/val)
        :return: processed model outputs
        """
        # Stack batch and restrict to generation length
        images = torch.stack([b['image'] for b in batch[0]])
        states = torch.stack([b['x'] for b in batch[0]]).squeeze(1)
        labels = torch.stack([b['class_id'] for b in batch[0]])

        # Same random portion of the sequence over generation_len, saving room for backwards solving
        random_start = np.random.randint(generation_len, images.shape[1] - self.args.z_amort - generation_len)

        # Get forward sequences
        images = images[:, random_start:random_start + generation_len + self.args.z_amort]
        states = states[:, random_start:random_start + generation_len + self.args.z_amort]

        # Get backwards sequence
        images_rev = torch.flip(images, dims=[1])

        # Get predictions
        preds, embeddings = self(images, generation_len)

        # Restrict images to start from after inference, for metrics and likelihood
        images = images[:, self.args.z_amort:]
        states = states[:, self.args.z_amort:]
        return images, images_rev, states, labels, preds, embeddings

    def get_step_losses(self, images, images_rev, preds):
        """
        Handles getting the ELBO terms for the given step
        :param images: ground truth images
        :param images_rev: grouth truth images, reversed for some models' secondary TRS loss
        :param preds: forward predictions from the model
        :return: likelihood, kl on z0, model-specific dynamics loss
        """
        # Reconstruction loss for the sequence and z0
        likelihood = self.reconstruction_loss(preds, images).sum([2, 3]).flatten().mean()

        # Initial encoder loss, KL[q(z_K|x_0:K) || p(z_K)]
        klz = self.encoder.kl_z_term()

        # Get the loss terms from the specific latent dynamics loss
        dynamics_loss = self.model_specific_loss(images, images_rev, preds)
        return likelihood, klz, dynamics_loss

    def training_step(self, batch, batch_idx):
        """
        PyTorch-Lightning training step where the network is propagated and returns a single loss value,
        which is automatically handled for the backward update
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get the generation length - either fixed or random between [1,T] depending on flags
        generation_len = np.random.randint(1, self.args.generation_len) if self.args.generation_varying is True else self.args.generation_len

        # Get model outputs from batch
        images, images_rev, states, labels, preds, embeddings = self.get_step_outputs(batch, generation_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, images_rev, preds)

        # Log ELBO loss terms
        self.log("likelihood", likelihood, prog_bar=True)
        self.log("klz_loss", klz, prog_bar=True)
        self.log("dynamics_loss", dynamics_loss)

        # Log various metrics
        self.log("train_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_pixel_mse", self.reconstruction_loss(preds, images).mean([1, 2, 3]).mean(), prog_bar=True, on_step=True)
        self.log("train_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'], prog_bar=False, on_step=True)

        # Determine KL annealing factor for the current step
        kl_factor = determine_annealing_factor(self.n_updates, anneal_update=1000)
        self.log('kl_factor', kl_factor, prog_bar=False)

        # Build the full loss
        loss = likelihood + kl_factor * ((self.args.z0_beta * klz) + dynamics_loss)

        # Return outputs as dict
        self.n_updates += 1
        out = {"loss": loss, "labels": labels.detach()}
        if batch_idx < self.args.batches_to_save:
            out["preds"] = preds.detach()
            out["images"] = images.detach()

            # For code vector based models (i.e. the proposed models) also add their local codes
            if hasattr(self.dynamics_func, 'embeddings'):
                out['code_vectors'] = self.dynamics_func.embeddings.detach()
        return out

    def training_epoch_end(self, outputs):
        """
        # Every 4 epochs, get a reconstruction example, model-specific plots, and copy over to the experiments folder
        :param outputs: list of outputs from the training steps, with the last 25 steps having reconstructions
        """
        if self.current_epoch % 10 != 0:
            return

        # Show side-by-side reconstructions
        show_images(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}train.png', num_out=5)

        # Get per-dynamics plots
        self.model_specific_plotting(self.version_path, outputs)

        # Copy experiment to relevant folder
        if self.args.exptype is not None:
            shutil.copytree(
                self.version_path, f"experiments/{self.args.exptype}/{self.args.model}/version_{self.exptop}",
                dirs_exist_ok=True
            )

    def validation_step(self, batch, batch_idx):
        """
        PyTorch-Lightning validation step. Similar to the training step but on the given val set under torch.no_grad()
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        images, images_rev, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_validation_len)

        # Get model loss terms for the step
        likelihood, klz, dynamics_loss = self.get_step_losses(images, images_rev, preds)
        del images_rev, states, labels, embeddings

        # Log validation likelihood and metrics
        self.log("val_likelihood", likelihood, prog_bar=True)
        self.log("val_vpt", vpt(images, preds.detach())[0], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_dst", dst(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_vpd", vpd(images, preds.detach())[1], prog_bar=True, on_epoch=True, on_step=False)

        # Get extrapolation metrics if validation generation is longer than training
        if self.args.generation_validation_len > self.args.generation_len:
            self.log("val_mse_recon", self.reconstruction_loss(preds[:, :self.args.generation_len],
                                                               images[:, :self.args.generation_len]).mean([1, 2, 3]).mean(),
                     prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_mse_extrapolation", self.reconstruction_loss(preds[:, self.args.generation_len:],
                                                                       images[:, self.args.generation_len:]).mean([1, 2, 3]).mean(),
                     prog_bar=True, on_epoch=True, on_step=False)
        else:
            self.log("val_mse_recon", self.reconstruction_loss(preds, images).mean([1, 2, 3]).mean(),
                     prog_bar=True, on_epoch=True, on_step=False)

        # Build the full loss
        loss = likelihood + dynamics_loss

        # Return outputs as dict
        out = {"loss": loss}
        if batch_idx == 0:
            out["preds"] = preds.detach()
            out["images"] = images.detach()
        return out

    def validation_epoch_end(self, outputs):
        """
        Every N epochs, get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        show_images(outputs[0]["images"], outputs[0]["preds"],
                    f'{self.version_path}/images/recon{self.current_epoch}val.png', num_out=5)

    def test_step(self, batch, batch_idx):
        """
        PyTorch-Lightning testing step.
        :param batch: list of dictionary objects representing a single image
        :param batch_idx: how far in the epoch this batch is
        """
        # Get model outputs from batch
        # TODO - Output N runs per batch to get averaged metrics rather than one run
        images, images_rev, states, labels, preds, embeddings = self.get_step_outputs(batch, self.args.generation_len)
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

        # Only get metrics on unseen part in generation
        preds = preds[:, self.args.training_len:]
        images = images[:, self.args.training_len:]
        states = states[:, self.args.training_len:]
        embeddings = embeddings[:, self.args.training_len:]

        # Print statistics over the full set
        pixel_mse = self.reconstruction_loss(preds, images).detach().cpu().numpy()
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

        # Set up output path and create dir
        output_path = f"{self.args.ckpt_path}/test_{self.args.dataset}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Save files
        np.save(f"{output_path}/test_{self.args.dataset}_pixelmse.npy", pixel_mse)
        np.save(f"{output_path}/test_{self.args.dataset}_recons.npy", preds.detach().cpu().numpy())
        np.save(f"{output_path}/test_{self.args.dataset}_images.npy", images.detach().cpu().numpy())

        # Save some examples
        show_images(images[:10], preds[:10], f"{output_path}/test_{self.args.dataset}_examples.png", num_out=5)

        # Save trajectory examples
        get_embedding_trajectories(embeddings[0], states[0], f"{output_path}/")

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.args.dataset}_metrics.json", 'w') as f:
            json.dump(metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.args.dataset}_excel.txt", 'w') as f:
            f.write(f"{metrics['mse_mean']},{metrics['mse_std']},{metrics['vpt_mean']},{metrics['vpt_std']},"
                    f"{metrics['dst_mean']},{metrics['dst_std']},{metrics['vpd_mean']},{metrics['vpd_std']}")
