"""
@file RGN.py

Holds the model for the Recurrent Generative Network latent dynamics function
"""
import torch
import torch.nn as nn

from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel


class RecurrentDynamicsFunction(nn.Module):
    def __init__(self, args):
        """ Latent dynamics function where the state is given and the next state is output """
        super(RecurrentDynamicsFunction, self).__init__()

        # Array that holds dimensions over hidden layers
        self.layers_dim = [args.latent_dim] + args.num_layers * [args.num_hidden] + [args.latent_dim]

        # Build network layers
        self.acts = []
        self.layers = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.gpus[0]))
            self.layer_norms.append(nn.LayerNorm(n_out, device=args.gpus[0]) if True and i < args.num_layers else nn.Identity())

    def forward(self, z):
        """ Given a latent state z, output z+1 """
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            z = a(norm(layer(z)))
        return z


class RGN(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic recurrent state function """
        super().__init__(args, top, exptop, last_train_idx)

        # Recurrent dynamics function
        self.dynamics_func = RecurrentDynamicsFunction(args)

    def forward(self, x):
        """
        Forward function of the RGN SSM model
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        """
        # Reshape images to combine generation_len and channels
        generation_len = x.shape[1]

        # Sample z0
        z = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, generation_len, generation_len).to(self.device)

        # Evaluate forward over timesteps by recurrently passing in output states
        zts = []
        for _ in t:
            z = self.dynamics_func(z)
            zts.append(z)

        # Stack zt and decode zts
        zt = torch.stack(zts)
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        parser = parent_parser.add_argument_group("RGN")
        parser.add_argument('--model_file', type=str, default="RGN", help='filename of the model')
        return parent_parser
