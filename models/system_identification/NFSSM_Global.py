"""
@file NFSSM_Global.py

PyTorch Lightning Implementation for the global hyperprior SSM,
which has a latent embedding space put on the neural units of the dynamics function
"""
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint
from models.CommonDynamics import LatentDynamicsModel
from utils.layers import GroupSwish, GroupTanh


class ODE(nn.Module):
    def __init__(self, args):
        """
        Represents the MetaPrior in the Global case where a single set of distributional parameters
        are optimized in the metaspace.
        :param args: script arguments to use for initialization
        """
        super(ODE, self).__init__()

        # Parameters
        self.args = args
        self.latent_dim = args.latent_dim
        self.hyper_dim = args.hyper_dim
        self.code_dim = args.code_dim
        self.conv_dim = args.num_filt * 4 ** 2
        self.n_groups = self.args.batch_size

        # Array that holds dimensions over hidden layers
        self.layers_dim = [self.latent_dim] + args.num_layers * [args.num_hidden] + [self.latent_dim]

        """ Hyper Prior for ODE """
        # Build code matrix
        self.codes = torch.nn.Parameter(0.01 * torch.randn([sum(self.layers_dim), self.code_dim], requires_grad=True).float())

        # Hyperprior function to convert weight codes into weights
        self.hyperprior = nn.Sequential(
            nn.Linear(self.code_dim * 2, self.hyper_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hyper_dim, 1),
        )

        # Build the grouped convolution dynamics network
        self.dynamics_network = nn.Sequential(
            nn.Conv1d(self.args.latent_dim * self.n_groups, self.args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupSwish(self.n_groups),
            nn.Conv1d(self.args.num_hidden * self.n_groups, self.args.num_hidden * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupSwish(self.n_groups),
            nn.Conv1d(self.args.num_hidden * self.n_groups, self.args.latent_dim * self.n_groups, 1, groups=self.n_groups, bias=True),
            GroupTanh(self.n_groups)
        )

        if self.args.ckpt_path == "None":
            self.init_parameters()

    def init_parameters(self):
        """
        Handles initializing the parameters of the hyper-network through the recommendations of
        "Hypernetworks in Meta-Reinforcement Learning"
        """
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.hyperprior[0].weight)
        std = 1. / math.sqrt(fan_in)
        a = 4 * math.sqrt(3.0) * std

        nn.init._no_grad_uniform_(self.hyperprior[0].weight, -a, a)
        nn.init.zeros_(self.hyperprior[0].bias)

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.hyperprior[-1].weight)
        std = 1. / math.sqrt(fan_in)
        a = 4 * math.sqrt(3.0) * std

        nn.init._no_grad_uniform_(self.hyperprior[-1].weight, -a, a)
        nn.init.zeros_(self.hyperprior[-1].bias)

    def sample_weights(self):
        # Reshape the vector into the sub-network layers
        codes = []
        next_idx = 0
        for i in range(len(self.layers_dim)):
            cur_idx = next_idx
            next_idx += self.layers_dim[i]
            codes.append(self.codes[cur_idx:next_idx].unsqueeze(0).repeat(self.args.batch_size, 1, 1))

        # Generate weight codes
        ws, bs = [], []
        for idx in range(len(self.layers_dim) - 1):
            # Construct the weight codes as the concatenation between all the neurons in subsequent layers
            weight_code = torch.cat((
                codes[idx + 1].unsqueeze(1).repeat(1, self.layers_dim[idx], 1, 1).view([-1, self.code_dim]),
                codes[idx].unsqueeze(2).repeat(1, 1, self.layers_dim[idx + 1], 1).view([-1, self.code_dim])
            ), dim=1)

            # Generate bias codes (concatenation is just with a zeros vector)
            bias_code = torch.cat((0.01 * torch.ones_like(codes[idx + 1]), codes[idx + 1]), dim=2).view([-1, self.code_dim * 2])

            # Get weights and biases out
            w = self.hyperprior(weight_code).view([self.args.batch_size * self.layers_dim[idx + 1], self.layers_dim[idx], 1])
            b = self.hyperprior(bias_code).view([self.args.batch_size * self.layers_dim[idx + 1], ])
            ws.append(w)
            bs.append(b)

            # Copy over the generated weights into the parameters of the dynamics network
            if hasattr(self.dynamics_network[idx * 2], 'weight'):
                del self.dynamics_network[idx * 2].weight
            self.dynamics_network[idx * 2].weight = w

            if hasattr(self.dynamics_network[idx * 2], 'bias'):
                del self.dynamics_network[idx * 2].bias
            self.dynamics_network[idx * 2].bias = b

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return self.dynamics_network(z)


class NFSSM_Global(LatentDynamicsModel):
    def __init__(self, args, top, exptop):
        super().__init__(args, top, exptop)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(args)

    def forward(self, x, generation_len):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for each sample.
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        :param generation_len: how many timesteps to generate over
        :return: reconstructions of the trajectory and generation
        """
        # Sample z_init
        z_init = self.encoder(x).reshape([1, -1, 1])

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, generation_len - 1, generation_len).to(self.device)

        # Draw weights from codes
        self.dynamics_func.sample_weights()

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method=self.args.integrator, options=dict(self.args.integrator_params))
        zt = zt.reshape([generation_len, self.args.batch_size, self.args.latent_dim])
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_plotting(self, version_path, outputs):
        """ Plots the mean of the MetaPrior codes in a shared space """
        # Plot prototypes
        next_idx = 0
        for i in range(len(self.dynamics_func.layers_dim)):
            cur_idx = next_idx
            next_idx += self.dynamics_func.layers_dim[i]
            plt.scatter(self.dynamics_func.codes[cur_idx:next_idx, 0].detach().cpu().numpy(),
                        self.dynamics_func.codes[cur_idx:next_idx, 1].detach().cpu().numpy())
        plt.legend([f"Layer {i}" for i in range(len(self.dynamics_func.layers_dim))])
        plt.title("t-SNE Plot of Code Embeddings")
        plt.savefig(f"lightning_logs/version_{self.top}/images/recon{self.current_epoch}tsne.png")
        plt.close()

    @staticmethod
    def get_model_specific_args():
        """ Add the hyperprior's arguments """
        return {
            "code_dim": 32,     # int (default=32) dimension of the weight codes
            "hyper_dim": 512    # int (default=512) dimension of the hyperprior function
        }
