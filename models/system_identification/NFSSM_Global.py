"""
@file NFSSM_Global.py

PyTorch Lightning Implementation for the global hyperprior SSM,
which has a latent embedding space put on the neural units of the dynamics function
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.utils import get_act
from torchdiffeq import odeint
from models.CommonDynamics import LatentDynamicsModel
from torch.distributions import Normal, kl_divergence as kl


class ODEFunc(nn.Module):
    def __init__(self, args):
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters
        are optimized in the metaspace.
        :param args: script arguments to use for initialization
        """
        super(ODEFunc, self).__init__()

        # Parameters
        self.args = args
        self.latent_dim = args.latent_dim
        self.hyper_dim = args.hyper_dim
        self.code_dim = args.code_dim

        # Array that holds dimensions over hidden layers
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]

        # Build activation layers
        self.acts = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(self.args.latent_act) if i < args.num_layers else get_act('linear'))
            self.layer_norms.append(nn.LayerNorm(n_out).to(self.args.gpus[0]) if True and i < args.num_layers else nn.Identity())

        # Build code priors
        self.code_mat_mus = nn.ParameterList([])
        self.code_mat_vars = nn.ParameterList([])
        for lidx in range(len(self.layers_dim)):
            # Mean
            code = torch.nn.Parameter(
                # 0.05 + (0.1 * torch.randn([self.layers_dim[lidx], self.code_dim])),
                torch.zeros([self.layers_dim[lidx], self.code_dim]),
                requires_grad=True
            )
            self.code_mat_mus.append(code)

            # Logvar
            code = torch.nn.Parameter(
                0.1 * torch.ones([self.layers_dim[lidx], self.code_dim]),
                requires_grad=True
            )
            self.code_mat_vars.append(code)

        # Hyperprior function to convert weight codes into weights
        self.hyperprior = nn.Sequential(
            nn.Linear(self.code_dim * 2, self.hyper_dim),
            nn.BatchNorm1d(self.hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.BatchNorm1d(self.hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hyper_dim, 1),
        )

        if self.args.checkpt == "None":
            self.reset_parameters()

    def reset_parameters(self):
        """ Smart initialization function for the distributional parameters of the meta-prior """
        for lidx in range(len(self.layers_dim)):
            # Initialization method of Adv-BNN
            stdv = 1. / math.sqrt(self.code_mat_mus[lidx].size(1))
            self.code_mat_mus[lidx].data.uniform_(-stdv, stdv)
            self.code_mat_vars[lidx].data.fill_(math.log(0.1))

        # Apply to the hyperprior funciton g(w|c)
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
        return kl(q, N).sum([-1]).mean()

    def sample_weights(self):
        """
        Before each ODE solution, weights need to be sample for the ODE-Net by generating current weight codes
        from the meta-space and sampling
        """
        # Sample the codes array
        self.codes = [
            self.code_mat_mus[i] + torch.randn_like(self.code_mat_mus[i]) * torch.exp(0.5 * self.code_mat_vars[i])
            for i in range(len(self.layers_dim))
        ]

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
            w = self.hyperprior(weight_code).view([self.layers_dim[idx], self.layers_dim[idx + 1]])
            b = self.hyperprior(bias_code).squeeze()
            self.w.append(w)
            self.b.append(b)

    def forward(self, t, x):
        """ Wrapper function for the odeint calculation """
        for norm, a, w, b in zip(self.layer_norms, self.acts, self.w, self.b):
            x = a(norm(F.linear(x, w.T, b)))
        return x


class NFSSM_Global(LatentDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunc(args)

    def forward(self, x, generation_len):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for each sample.
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        :return: reconstructions of the trajectory and generation
        """
        # Sample z_init
        z_init = self.encoder(x)

        # Draw function for index
        self.dynamics_func.sample_weights()

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method='rk4', options={'step_size': 0.125})

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_loss(self, x, x_rec, x_rev, train=True):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        # Get KL C loss
        klc_loss = self.dynamics_func.kl_c_term()
        dynamics_loss = (self.args.kl_beta * klc_loss)

        # Log and return
        if train:
            self.log("klc_loss", self.args.kl_beta * klc_loss, prog_bar=True, on_step=True, on_epoch=False)
        return dynamics_loss

    def model_specific_plotting(self, version_path, outputs):
        """ Plots the mean of the MetaPrior codes in a shared space """
        for c in self.dynamics_func.code_mat_mus:
            plt.scatter(c[:, 0].detach().cpu().numpy(), c[:, 1].detach().cpu().numpy())

        plt.legend([f'Layer{i}' for i in range(len(self.dynamics_func.layers_dim))])
        plt.title(f"Meta-variables Mean @ Epoch {self.current_epoch}")
        plt.savefig(f'lightning_logs/version_{self.top}/images/recon{self.current_epoch}meta.png')
        plt.close()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Add the hyperprior's arguments  """
        parser = parent_parser.add_argument_group("NFSSM_Global")

        # Network dimensions
        parser.add_argument('--code_dim', type=int, default=32, help='dimension of the weight codes')
        parser.add_argument('--hyper_dim', type=int, default=512, help='dimension of the hyperprior function')
        return parent_parser
