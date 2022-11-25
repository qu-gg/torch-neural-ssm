"""
@file NeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
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
            self.acts.append(get_act(args.latent_act) if i < args.num_layers else get_act('tanh'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.gpus[0]))
            self.layer_norms.append(nn.LayerNorm(n_out, device=args.gpus[0]) if True and i < args.num_layers else nn.Identity())

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, layer in zip(self.layer_norms, self.acts, self.layers):
            x = a(norm(layer(x)))
        return x


class NeuralODE(LatentDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t,
                    method=self.args.integrator, options=dict(self.args.integrator_params)
                    )  # [T,q]
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt
