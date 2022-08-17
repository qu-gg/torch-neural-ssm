"""
@file NeuralODE_SI.py

Holds the model for the Neural ODE latent dynamics function
"""
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint

from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        """ Standard Neural ODE dynamics function """
        super(ODEFunction, self).__init__()

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.gpus[0]))
            self.layer_norms.append(nn.LayerNorm(n_out, device=args.gpus[0]) if True and i < args.num_layers else nn.Identity())

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            x = a(norm(layer(x)))
        return x


class NeuralODE_SI(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop, last_train_idx)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, **kwargs):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        """
        # Reshape images to combine generation_len and channels
        generation_len = x.shape[1]

        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len, generation_len + 1).to(self.device)

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.5})  # [T,q]
        zt = zt.permute([1, 0, 2])[:, 1:]

        # Stack zt and decode zts
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        parser = parent_parser.add_argument_group("NODE")
        parser.add_argument('--model_file', type=str, default="system_identification/NeuralODE_SI", help='filename of the model')
        return parent_parser
