"""
@file CommonVAE.py

Holds the encoder/decoder architectures that are shared across the NSSM works
"""
import torch
import torch.nn as nn

from utils.layers import Flatten, Gaussian, UnFlatten
from torch.distributions import Normal, kl_divergence as kl


class LatentStateEncoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional encoder that takes in a sequence of images and outputs the
        initial state of the latent dynamics
        :param args: argparse dictionary
        """
        super(LatentStateEncoder, self).__init__()
        self.args = args

        # Encoder, q(z_0 | x_{0:args.z_amort})
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(args.z_amort, args.num_filters, kernel_size=5, stride=2, padding=(2, 2)),  # 14,14
            nn.BatchNorm2d(args.num_filters),
            nn.LeakyReLU(),
            nn.Conv2d(args.num_filters, args.num_filters * 2, kernel_size=5, stride=2, padding=(2, 2)),  # 7,7
            nn.BatchNorm2d(args.num_filters * 2),
            nn.LeakyReLU(),
            nn.Conv2d(args.num_filters * 2, args.num_filters * 4, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(args.num_filters * 4),
            nn.LeakyReLU(),
            nn.AvgPool2d(4),
            Flatten()
        )

        self.initial_encoder_out = Gaussian(args.num_filters * 4, args.latent_dim) if args.stochastic is True \
            else nn.Linear(args.num_filters * 4, args.latent_dim)

        self.out_act = nn.Tanh()

        # Holds generated z0 means and logvars for use in KL calculations
        self.z_means = None
        self.z_logvs = None

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        if self.args.stochastic is False:
            return 0.0

        batch_size = self.z_means.shape[0]
        mus, logvars = self.z_means.view([-1]), self.z_logvs.view([-1])  # N, 2

        q = Normal(mus, torch.exp(0.5 * logvars))
        N = Normal(torch.zeros(len(mus), device=mus.device),
                   torch.ones(len(mus), device=mus.device))

        klz = kl(q, N).view([batch_size, -1]).sum([1]).mean()
        return klz

    def forward(self, x):
        """
        Handles getting the initial state given x and saving the distributional parameters
        :param x: input sequences [BatchSize, GenerationLen * NumChannels, H, W]
        :return: z0 over the batch [BatchSize, LatentDim]
        """
        if self.args.stochastic is True:
            self.z_means, self.z_logvs, z0 = self.initial_encoder_out(self.initial_encoder(x[:, :self.args.z_amort]))
        else:
            z0 = self.initial_encoder_out(self.initial_encoder(x[:, :self.args.z_amort]))
        return self.out_act(z0)


class EmissionDecoder(nn.Module):
    def __init__(self, args):
        """
        Holds the convolutional decoder that takes in a batch of individual latent states and
        transforms them into their corresponding data space reconstructions
        """
        super(EmissionDecoder, self).__init__()
        self.args = args

        # Variable that holds the estimated output for the flattened convolution vector
        self.conv_dim = args.num_filters * 4 ** 3

        # Emission model handling z_i -> x_i
        self.decoder = nn.Sequential(
            # Transform latent vector into 4D tensor for deconvolution
            nn.Linear(args.latent_dim, self.conv_dim),
            nn.BatchNorm1d(self.conv_dim),
            nn.LeakyReLU(),
            UnFlatten(4),

            # Perform de-conv to output space
            nn.ConvTranspose2d(self.conv_dim // 16, args.num_filters * 4, kernel_size=4, stride=1, padding=(0, 0)),
            nn.BatchNorm2d(args.num_filters * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(args.num_filters * 4, args.num_filters * 2, kernel_size=5, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(args.num_filters * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(args.num_filters * 2, args.num_filters, kernel_size=5, stride=2, padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(args.num_filters),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(args.num_filters, args.num_channels, kernel_size=5, stride=1, padding=(2, 2)),
            nn.Sigmoid(),
        )

    def forward(self, zts):
        """
        Handles decoding a batch of individual latent states into their corresponding data space reconstructions
        :param zts: latent states [BatchSize * GenerationLen, LatentDim]
        :return: data output [BatchSize, GenerationLen, NumChannels, H, W]
        """
        # Decode back to image space, after first flattening BatchSize * SeqLen
        x_rec = self.decoder(zts.contiguous().view([zts.shape[0] * zts.shape[1], -1]))

        # Reshape to image output
        x_rec = x_rec.view([zts.shape[0], x_rec.shape[0] // zts.shape[0], self.args.dim, self.args.dim])
        return x_rec
