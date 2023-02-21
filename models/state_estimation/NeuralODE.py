"""
@file NeuralODE_SE.py

Holds the state estimation model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel
from models.CommonVAE import LatentStateEncoder


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


class NeuralODE(LatentDynamicsModel):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.ode_func = ODEFunction(args)

        # Observation encoder
        self.obs_encoder = LatentStateEncoder(1, args.num_filt, 1, args.latent_dim, args.fix_variance)

        # Correction cell
        self.correction = nn.GRUCell(input_size=args.latent_dim, hidden_size=args.latent_dim)

    def forward(self, x, generation_len):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, generation_len - 1, generation_len - 1).to(self.device)

        zt = [z_init]
        prev_z = z_init
        for tidx in t:
            # Propagate forward one timestep with ODE
            z_pred = odeint(self.ode_func, prev_z, torch.Tensor([0, 1]),
                            method=self.args.integrator, options=dict(self.args.integrator_params)
                            )[1]

            # Encode observation at step
            z_obs = self.obs_encoder(x[:, int(tidx) - 1].unsqueeze(1))

            # Correction prediction with encoding
            prev_z = self.correction(z_pred, z_obs)
            zt.append(prev_z)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt
