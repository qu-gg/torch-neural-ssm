"""
@file LSTM.py

Holds the model for the LSTM latent dynamics function
"""
import torch
import torch.nn as nn
from models.CommonDynamics import Dynamics


class LSTM(Dynamics):
    def __init__(self, args, top, exptop):
        """ Latent dynamics as parameterized by a global deterministic LSTM """
        super(Dynamics).__init__(args, top, exptop)

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

            z = self.dynamics_out(z_hid)
            zt.append(z)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        zt = zt.contiguous().view([zt.shape[0] * zt.shape[1], -1])
        x_rec = self.decoder(zt)
        return x_rec

    def model_specific_loss(self):
        # No specific model loss
        return 0
