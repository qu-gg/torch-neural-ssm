"""
@file DVBF.py

Holds the model for the Deep Variational Bayes Filter baseline, source code modified from
@url{https://github.com/gregorsemmler/pytorch-dvbf}

Instead of having access to GT observations at all timesteps, it gets access to the first N observations
before sampling on the w matrix stems from the prior itself (as described in the original work)
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from models.CommonDynamics import LatentDynamicsModel
from torch.distributions import Normal, kl_divergence as kl


def get_output_shape(layer, shape):
    layer_training = layer.training
    if layer_training:
        layer.eval()
    out = layer(torch.zeros(1, *shape))
    before_flattening = tuple(out.size())[1:]
    after_flattening = int(np.prod(out.size()))
    if layer_training:
        layer.train()
    return before_flattening, after_flattening


def build_basic_encoder(num_in_channels=3, hidden_channels=(32, 64, 128, 256), kernel_size=3, stride=2, padding=1):
    layers = []
    in_c = num_in_channels
    for h_dim in hidden_channels:
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_c, out_channels=h_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_c = h_dim

    return nn.Sequential(*layers)


def build_basic_decoder(num_out_channels=3, decoder_filters=(512, 256, 128, 64, 32), kernel_size=3, stride=2, padding=1, output_padding=1):
    layers = []
    for i in range(len(decoder_filters) - 1):
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_filters[i], decoder_filters[i + 1], kernel_size=kernel_size,
                                   stride=stride, padding=padding, output_padding=output_padding),
                nn.BatchNorm2d(decoder_filters[i + 1]),
                nn.LeakyReLU())
        )

    layers.append(nn.ConvTranspose2d(decoder_filters[-1], num_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class FakeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def kl_z_term(self):
        return torch.Tensor([0.]).to(self.args.gpus[0])


class DVBF(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop, last_train_idx)

        self.encoder = FakeEncoder(args)

        self.input_shape = (1, self.args.dim, self.args.dim)
        self.latent_dim = self.args.latent_dim
        self.action_dim = 0
        self.use_actions = self.action_dim > 0
        self.embedder_filters = (32, 64, 128, 256, 512)
        self.embedder = build_basic_encoder(1, self.embedder_filters)
        self.decoder = build_basic_decoder(1, list(reversed(self.embedder_filters)))
        self.action_encoder = lambda x: x
        encoder_out_size, encoder_out_size_flat = get_output_shape(self.embedder, self.input_shape)

        self.decoder_input_size = encoder_out_size
        self.decoder_input_size_flat = encoder_out_size_flat

        self.w_mu = nn.Linear(encoder_out_size_flat, self.args.latent_dim)
        self.w_log_var = nn.Linear(encoder_out_size_flat, self.args.latent_dim)

        self.decoder_input = nn.Linear(self.args.latent_dim, self.decoder_input_size_flat)
        self.num_matrices = 15
        self.a_i_list = []
        self.b_i_list = []
        self.c_i_list = []
        self.initialize_matrices()

        self.weight_network_hidden_size = 64
        self.matrix_weight_network = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.weight_network_hidden_size),
            nn.ELU(),
            nn.Linear(self.weight_network_hidden_size, self.num_matrices),
            nn.Softmax(dim=-1)
        )

        self.initial_hidden_size = self.latent_dim
        self.initial_z_model = nn.Sequential(
            nn.Linear(self.latent_dim, self.initial_hidden_size),
            nn.ELU(),
            nn.Linear(self.initial_hidden_size, self.latent_dim)
        )

        self.num_rnn_layers = 1
        self.num_rnn_directions = 2
        self.to_initial_w_in = nn.Linear(encoder_out_size_flat, self.latent_dim)
        self.initial_w_model = nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim,
                                       num_layers= self.num_rnn_layers, bidirectional=True)
        self.w_initial_mu = nn.Linear(self.num_rnn_directions * self.latent_dim, self.latent_dim)
        self.w_initial_log_var = nn.Linear(self.num_rnn_directions * self.latent_dim, self.latent_dim)

    def initialize_matrices(self):
        for idx in range(self.num_matrices):
            a_m = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
            init.kaiming_uniform_(a_m)
            self.register_parameter(f"A_{idx}", a_m)
            self.a_i_list.append(a_m)

            if self.use_actions:
                b_m = nn.Parameter(torch.Tensor(self.latent_dim, self.action_dim))
                init.kaiming_uniform_(b_m)
                self.register_parameter(f"B_{idx}", b_m)
                self.b_i_list.append(b_m)

            c_m = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
            self.register_parameter(f"C_{idx}", c_m)
            init.kaiming_uniform_(c_m)
            self.c_i_list.append(c_m)

    def get_w(self, encoder_out_flat):
        mu = self.w_mu(encoder_out_flat)
        log_var = self.w_log_var(encoder_out_flat)
        return self.sample(mu, log_var), mu, log_var

    def get_initial_w(self, encoder_out_flats):
        batch_size = encoder_out_flats[0].shape[0]
        rnn_ins = [self.to_initial_w_in(el) for el in encoder_out_flats]
        rnn_in = torch.stack(rnn_ins)

        _, (h_n, _) = self.initial_w_model(rnn_in)
        decoder_input_in = h_n.transpose(0, 1).contiguous()
        decoder_input_in = decoder_input_in.view(batch_size, self.num_rnn_layers, self.num_rnn_directions, -1)
        decoder_input_in = decoder_input_in[:, -1, :, :].view(batch_size, -1)  # Use only last layer

        w_mu = self.w_initial_mu(decoder_input_in)
        w_log_var = self.w_initial_log_var(decoder_input_in)

        w = self.sample(w_mu, w_log_var)
        return w, w_mu, w_log_var

    def get_initial_z(self, w):
        return self.initial_z_model(w)

    def sample(self, mu, log_var):
        std_dev = torch.exp(0.5 * log_var)
        return torch.randn_like(log_var) * std_dev + mu

    def get_next_z(self, z, w, u=None):
        model_in = z
        alpha = self.matrix_weight_network(model_in)

        next_z_list = []
        for b_idx in range(alpha.shape[0]):
            curr_z = z[b_idx]

            a_matrix = torch.sum(torch.stack([alpha[b_idx, j] * self.a_i_list[j] for j in range(self.num_matrices)]), dim=0)
            c_matrix = torch.sum(torch.stack([alpha[b_idx, j] * self.c_i_list[j] for j in range(self.num_matrices)]), dim=0)

            next_z = a_matrix.matmul(curr_z) + c_matrix.matmul(w[b_idx])
            next_z_list.append(next_z)
        return torch.stack(next_z_list)

    def simulate_next(self, z, u, device, w=None):
        if w is None:
            w = torch.randn((1, self.latent_dim)).to(device)
        z = self.get_next_z(z, w, u)
        return self.decoder(self.decoder_input(z).view((-1,) + self.decoder_input_size)), z

    def forward(self, x, **kwargs):
        """
        Forward function of the network that handles locally embedding the given sample into the C codes,
        generating the z posterior that defines mixture weightings, and finding the winning components for
        each sample
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        """
        # Add in channels dimension for [BS, SL, 1, Dim, Dim]
        x = x.unsqueeze(2)

        # Permute to [SL, BS, 1, Dim, Dim]
        x = x.permute(1, 0, 2, 3, 4)
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Get the encoded variables for the given sequences
        encoder_outs = [self.embedder(x[idx, :, :, :, :]) for idx in range(self.args.z_amort)]
        encoder_out_flats = [el.view(batch_size, -1) for el in encoder_outs]

        # Get initial w parameters and initial latent state
        initial_w, initial_w_mu, initial_w_log_var = self.get_initial_w(encoder_out_flats)
        initial_z = self.get_initial_z(initial_w)

        # For the observed sequence, get the next W from the RNN encoder and get the next-step prediction
        self.w_mus = [initial_w_mu]
        self.w_log_vars = [initial_w_log_var]
        z_s = [initial_z]
        w_s = [initial_w]
        for seq_idx in range(1, seq_len):
            # Reconstruction
            if seq_idx < self.args.z_amort:
                encoder_out_flat = encoder_out_flats[seq_idx]
                prev_z = z_s[seq_idx - 1]
                prev_u = None
                prev_w = w_s[-1]
                w, w_mu, w_log_var = self.get_w(encoder_out_flat)

                self.w_mus.append(w_mu)
                self.w_log_vars.append(w_log_var)
                w_s.append(w)
                z_s.append(self.get_next_z(prev_z, prev_w, prev_u))
            # Generation given 1) prior and 2) previous autoregressive z
            else:
                prev_z = z_s[seq_idx - 1]
                prev_w = w_s[-1]

                w = torch.randn((self.args.batch_size, self.latent_dim)).to(self.args.gpus[0])
                self.w_mus.append(torch.zeros_like(initial_w_mu).to(self.args.gpus[0]))
                self.w_log_vars.append(torch.full_like(initial_w_log_var, fill_value=0.1).to(self.args.gpus[0]))

                w_s.append(w)
                z_s.append(self.get_next_z(prev_z, prev_w, None))

        # Decode for output
        x_rec = [self.decoder(self.decoder_input(el).view((-1,) + self.decoder_input_size)) for el in z_s]
        x_rec = torch.stack(x_rec).squeeze(2)
        x_rec = x_rec.permute(1, 0, 2, 3)

        # Stack embeddings
        z_s = torch.stack(z_s, dim=1)
        return x_rec, z_s

    def model_specific_loss(self, x, x_rec, zts, train=True):
        """ KL term between the parameter distribution w and a normal prior"""
        w_mus = torch.stack(self.w_mus, dim=1)
        w_log_vars = torch.stack(self.w_log_vars, dim=1)

        w_mus = w_mus.reshape([w_mus.shape[0], -1])
        w_log_vars = w_log_vars.reshape([w_log_vars.shape[0], -1])

        q = Normal(w_mus, torch.exp(0.5 * w_log_vars))
        N = Normal(torch.zeros_like(w_mus, device=w_mus.device),
                   torch.ones_like(w_mus, device=w_mus.device))
        return kl(q, N).sum([-1]).mean()

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser


