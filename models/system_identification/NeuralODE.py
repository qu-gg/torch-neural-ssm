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
        cfg_arch = cfg.model.architecture

        # Array that holds dimensions over hidden layers
        self.layers_dim = [cfg_arch.latent_dim] + cfg_arch.num_layers * [cfg_arch.num_hidden] + [cfg_arch.latent_dim]

        # Build network layers
        self.acts = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(cfg_arch.latent_act) if i < cfg_arch.num_layers else get_act('linear'))
            self.layers.append(nn.Linear(n_in, n_out, device=cfg.training.devices[0]))

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x


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
        if self.cfg.model.integrator == 'symplectic':
            # configure training options
            options = {'method': 'sym12async', 'h': None, 't0': 0.0, 't1': generation_len - 1, 't_eval': t,
                       'rtol': 1e-2, 'atol': 1e-4, 'print_neval': False, 'neval_max': 1000000, 'safety': None,
                       'interpolation_method': 'cubic', 'regenerate_graph': False, 'print_time': False}
            zt = odesolve(self.dynamics_func, z_init, options=options)  # [T,q]
            if len(zt.shape) == 2:
                zt = zt.unsqueeze(0)
        else:
            zt = odeint(self.dynamics_func, z_init, t, method=self.cfg.model.integrator, options=dict(self.cfg.model.integrator_params))
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt
