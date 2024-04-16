"""
@file NeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel
from TorchDiffEqPack.odesolver import odesolve


class ODEFunction(nn.Module):
    def __init__(self, cfg):
        """ Standard Neural ODE dynamics function """
        super(ODEFunction, self).__init__()

        # Build the dynamics network
        dynamics_network = []
        dynamics_network.extend([
            nn.Linear(cfg.architecture.latent_dim, cfg.architecture.num_hidden),
            get_act(cfg.architecture.latent_act)
        ])

        for _ in range(cfg.architecture.num_layers - 1):
            dynamics_network.extend([
                nn.Linear(cfg.architecture.num_hidden, cfg.architecture.num_hidden),
                get_act(cfg.architecture.latent_act)
            ])

        dynamics_network.extend([nn.Linear(cfg.architecture.num_hidden, cfg.architecture.latent_dim), nn.Tanh()])
        self.dynamics_network = nn.Sequential(*dynamics_network)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return self.dynamics_network(z)


class NeuralODE(LatentDynamicsModel):
    def __init__(self, cfg):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(cfg)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(cfg)

    def forward(self, x, generation_len):
        """
        Forward function of the ODE network
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len, device=self.device)

        # Perform the integration with the Neural ODE.
        # Depending on choice of integrator, different packages need to be used (i.e. symplectic is not in torchdiffeq)
        if self.cfg.integrator == 'symplectic':
            # configure training options
            options = {'method': 'sym12async', 'h': None, 't0': 0.0, 't1': generation_len - 1, 't_eval': t,
                       'rtol': 1e-2, 'atol': 1e-4, 'print_neval': False, 'neval_max': 1000000, 'safety': None,
                       'interpolation_method': 'cubic', 'regenerate_graph': False, 'print_time': False}
            zt = odesolve(self.dynamics_func, z_init, options=options)  # [T,q]
            if len(zt.shape) == 2:
                zt = zt.unsqueeze(0)
        else:
            zt = odeint(self.dynamics_func, z_init, t, method=self.cfg.integrator, options=dict(self.cfg.integrator_params))
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt
