"""
@file MetaODE_SI.py

Holds the model for the MetaPrior Netural ODE latent dynamics function
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from utils.utils import get_act
from utils.layers import Gaussian
from models.CommonDynamics import LatentDynamicsModel
from torch.distributions import Normal, kl_divergence as kl


class ODEFunction(nn.Module):
    def __init__(self, args):
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters
        are optimized in the metaspace.
        :param args: script arguments to use for initialization
        """
        super(ODEFunction, self).__init__()

        # Parameters
        self.args = args
        self.latent_dim = args.latent_dim
        self.hyper_dim = args.hyper_dim
        self.code_dim = args.code_dim

        # Array that holds dimensions over hidden layers
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]

        # Build activation layers
        self.acts = []
        self.layer_norms = []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act('leaky_relu') if i < args.num_layers else get_act('linear'))
            self.layer_norms.append(nn.LayerNorm(n_out, device=0) if True and i < args.num_layers else nn.Identity())

        # Build code priors
        self.codes = nn.ParameterList([])
        for lidx in range(len(self.layers_dim)):
            # Mean
            code = nn.Parameter(torch.zeros([self.layers_dim[lidx], self.code_dim]) + 1e-6)
            self.codes.append(code)

        # Hyperprior function to convert weight codes into weights
        self.hyperprior = nn.Sequential(
            nn.Linear(self.code_dim * 2, self.hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.LeakyReLU(0.1),
            Gaussian(self.hyper_dim, 1),
        )

        if self.args.checkpt == "None":
            self.reset_parameters()

    def reset_parameters(self):
        """ Smart initialization function for the distributional parameters of the meta-prior """
        self.hyper_layers = [self.code_dim * 2, self.hyper_dim, 1]
        lidx = 0
        for l in self.hyperprior:
            if isinstance(l, nn.Linear):
                w = l.weight
                b = l.bias

                fan_out, fan_in = w.size(0), w.size(1)

                bound = math.sqrt(3 * 2 / (fan_in * self.hyper_layers[lidx]) / 9)
                w.data.uniform_(-bound, bound)
                b.data.fill_(0)
                lidx += 1

    def kl_c_term(self):
        """ KL term between the latent codes and a standard Gaussian """
        mus = torch.cat([cmu.view([-1]) for cmu in self.code_mat_mus])
        logvar = torch.cat([cvar.view([-1]) for cvar in self.code_mat_vars])

        q = Normal(mus, torch.exp(0.5 * logvar))
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))
        return kl(q, N).sum()

    def sample_weights(self):
        """
        Before each ODE solution, weights need to be sample for the ODE-Net by generating current weight codes
        from the meta-space and sampling
        """
        # Generate weight codes
        self.w = []
        self.b = []
        for idx in range(len(self.layers_dim) - 1):
            temp = self.codes[idx].unsqueeze(1).repeat(1, self.layers_dim[idx + 1], 1).view([-1, self.code_dim])
            temp2 = self.codes[idx + 1].unsqueeze(0).repeat(self.layers_dim[idx], 1, 1).view([-1, self.code_dim])
            weight_code = torch.cat((temp2, temp), dim=1)

            # Generate bias codes (concatenation is just with a zeros vector)
            bias_code = torch.cat((torch.zeros_like(self.codes[idx + 1]), self.codes[idx + 1]), dim=1)

            # Get weights and biases out
            w = self.hyperprior(weight_code)[0].view([self.layers_dim[idx], self.layers_dim[idx + 1]])
            b = self.hyperprior(bias_code)[0].squeeze()

            self.w.append(w)
            self.b.append(b)

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.w, self.b):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class MetaODEDet_SI(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop, last_train_idx)

        # ODE-Net which holds mixture logic
        self.ode_func = ODEFunction(args)

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

        # Draw function for index
        self.ode_func.sample_weights()

        # Evaluate forward over timestep
        zt = odeint(self.ode_func, z_init, t, method='rk4', options={'step_size': 0.5})  # [T,q]
        zt = zt.permute([1, 0, 2])[:, 1:]

        # Stack zt and decode zts
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        parser = parent_parser.add_argument_group("NODE")
        parser.add_argument('--model_file', type=str, default="system_identification/MetaODEDet_SI", help='filename of the model')

        # Number of dimensions for the embedded neural unit vector
        parser.add_argument('--code_dim', type=int, default=8, help='dimension of the embedded neural unit vector')

        # Number of hidden units in the hyperprior function
        parser.add_argument('--hyper_dim', type=int, default=128, help='hidden units in the hyperprior function')
        return parent_parser
