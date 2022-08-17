"""
@file LSTM_SI.py

Holds the system identification model for the LSTM latent dynamics function
"""
import torch
import torch.nn as nn

from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel


class LSTM_SI(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic LSTM """
        super().__init__(args, top, exptop, last_train_idx)

        # Latent activation function
        self.latent_act = get_act(args.latent_act)

        # Recurrent dynamics function
        self.dynamics_func = nn.LSTMCell(input_size=args.latent_dim, hidden_size=args.num_hidden)
        self.dynamics_out = nn.Linear(args.num_hidden, args.latent_dim)

    def forward(self, x):
        """
        Forward function of the RGN SSM model
        :param x: data observation, which is a timeseries [BS, Timesteps, N Channels, Dim1, Dim2]
        """
        # Reshape images to combine generation_len and channels
        generation_len = x.shape[1]

        # Sample z0
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, generation_len, generation_len).to(self.device)

        # Evaluate forward over timesteps by recurrently passing in output states
        zt = []
        for tidx in t:
            if tidx == 1:
                z_hid, c_hid = self.dynamics_func(z_init)
            else:
                z_hid, c_hid = self.dynamics_func(z, (z_hid, c_hid))

            z_hid = self.latent_act(z_hid)
            z = self.dynamics_out(z_hid)
            zt.append(z)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Placeholder function for model-specific arguments """
        parser = parent_parser.add_argument_group("LSTM")
        parser.add_argument('--model_file', type=str, default="system_identification/LSTM", help='filename of the model')
        return parent_parser