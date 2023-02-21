"""
@file LSTM_SE.py

Holds the state estimatnion model for the LSTM latent dynamics function, in which a GRU Cell
is used to correct the latent prediction at each timestep
"""
import torch
import torch.nn as nn

from utils.utils import get_act
from models.CommonDynamics import LatentDynamicsModel
from models.CommonVAE import LatentStateEncoder


class LSTM_SE(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic LSTM """
        super().__init__(args, top, exptop, last_train_idx)

        # Latent activation function
        self.latent_act = get_act(args.latent_act)

        # Observation encoder
        self.obs_encoder = LatentStateEncoder(1, args.num_filt, 1, args.latent_dim, args.fix_variance)

        # Recurrent dynamics function
        self.dynamics_func = nn.LSTMCell(input_size=args.latent_dim, hidden_size=args.latent_dim)
        self.dynamics_out = nn.Linear(args.num_hidden, args.latent_dim)

        # Correction cell
        self.correction = nn.GRUCell(input_size=args.latent_dim, hidden_size=args.latent_dim)

    def forward(self, x, generation_len):
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
        z_pred = z_init
        c_pred = torch.zeros_like(z_pred)
        zt = []
        for tidx in t:
            # Encode observation at step
            z_obs = self.obs_encoder(x[:, int(tidx) - 1].unsqueeze(1))

            # Perform LSTM step
            z_pred, c_pred = self.dynamics_func(z_obs, (z_pred, c_pred))
            zt.append(z_pred)

        # Stack zt and decode zts
        zt = torch.stack(zt)
        x_rec = self.decoder(zt.contiguous().view([zt.shape[0] * zt.shape[1], -1]))
        return x_rec, zt
