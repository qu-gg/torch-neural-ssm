"""
@file NeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel


class RGNResFunction(nn.Module):
    def __init__(self, cfg):
        """ Standard Residual Recurrent Generative Network dynamics function """
        super(RGNResFunction, self).__init__()

        # Build the dynamics network
        dynamics_network = []
        dynamics_network.extend([
            nn.Linear(cfg.latent_dim, cfg.num_hidden),
            get_act(cfg.latent_act)
        ])

        for _ in range(cfg.num_layers - 1):
            dynamics_network.extend([
                nn.Linear(cfg.num_hidden, cfg.num_hidden),
                get_act(cfg.latent_act)
            ])

        dynamics_network.extend([nn.Linear(cfg.num_hidden, cfg.latent_dim), nn.Tanh()])
        self.dynamics_network = nn.Sequential(*dynamics_network)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return z + self.dynamics_network(z)


class RGNRes(LatentDynamicsModel):
    def __init__(self, cfg):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(cfg)

        # ODE-Net which holds mixture logic
        self.dynamics_func = RGNResFunction(cfg)

    def forward(self, x, generation_len):
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate forward over timestep
        z_cur = z_init
        zts = [z_init]
        for _ in range(generation_len - 1):
            z_cur = self.dynamics_func(None, z_cur)
            zts.append(z_cur)
        
        zt = torch.stack(zts, dim=1)

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt
