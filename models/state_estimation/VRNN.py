"""
@file VRNN.py

Holds the model for the Variational Recurrent Neural Network baseline, source code modified from
@url{https://github.com/XiaoyuBIE1994/DVAE/blob/master/dvae/model/vrnn.py}
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from models.CommonDynamics import LatentDynamicsModel
from torch.distributions import Normal, kl_divergence as kl


class FakeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def kl_z_term(self):
        return torch.Tensor([0.]).to(self.args.gpus[0])


class VRNN(LatentDynamicsModel):
    def __init__(self, args, top, exptop, last_train_idx):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args, top, exptop, last_train_idx)

        self.encoder = FakeEncoder(args)

        ### General parameters
        self.x_dim = self.args.dim ** 2
        self.z_dim = self.args.latent_dim
        self.dropout_p = 0.2
        self.y_dim = self.x_dim
        self.activation = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        ### Feature extractors
        self.dense_x = [512]
        self.dense_z = [512]

        ### Dense layers
        self.dense_hx_z = [256]
        self.dense_hz_x = [256]
        self.dense_h_z = [256]

        ### RNN
        self.dim_RNN = 128
        self.num_RNN = 2

        ### Beta-loss
        self.beta = 1

        ###########################
        #### Feature extractor ####
        ###########################
        # x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_feature_x = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_x[n - 1], self.dense_x[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_x = nn.Sequential(dic_layers)
        # z
        dic_layers = OrderedDict()
        if len(self.dense_z) == 0:
            dim_feature_z = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_z = self.dense_z[-1]
            for n in range(len(self.dense_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.z_dim, self.dense_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_z[n - 1], self.dense_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_z = nn.Sequential(dic_layers)

        ######################
        #### Dense layers ####
        ######################
        # 1. h_t, x_t to z_t (Inference)
        dic_layers = OrderedDict()
        if len(self.dense_hx_z) == 0:
            dim_hx_z = self.dim_RNN + dim_feature_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_z = self.dense_hx_z[-1]
            for n in range(len(self.dense_hx_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_x[-1] + self.dim_RNN, self.dense_hx_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_hx_z[n - 1], self.dense_hx_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_hx_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_hx_z, self.z_dim)

        # 2. h_t to z_t (Generation z)
        dic_layers = OrderedDict()
        if len(self.dense_h_z) == 0:
            dim_h_z = self.dim_RNN
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_z = self.dense_h_z[-1]
            for n in range(len(self.dense_h_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN, self.dense_h_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_h_z[n - 1], self.dense_h_z[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_z = nn.Sequential(dic_layers)
        self.prior_mean = nn.Linear(dim_h_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_h_z, self.z_dim)

        # 3. h_t, z_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN + dim_feature_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dim_RNN + dim_feature_z, self.dense_hz_x[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_hz_x[n - 1], self.dense_hz_x[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_hz_x, self.y_dim)

        ####################
        #### Recurrence ####
        ####################
        self.rnn = nn.LSTM(dim_feature_x + dim_feature_z, self.dim_RNN, self.num_RNN)

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return torch.addcmul(mean, eps, std)

    def generation_x(self, feature_zt, h_t):
        dec_input = torch.cat((feature_zt, h_t), 2)
        dec_output = self.mlp_hz_x(dec_input)
        y_t = self.gen_out(dec_output)
        y_t = self.sigmoid(y_t)
        return y_t

    def generation_z(self, h):
        prior_output = self.mlp_h_z(h)
        mean_prior = self.prior_mean(prior_output)
        logvar_prior = self.prior_logvar(prior_output)
        return mean_prior, logvar_prior

    def inference(self, feature_xt, h_t):
        enc_input = torch.cat((feature_xt, h_t), 2)
        enc_output = self.mlp_hx_z(enc_input)
        mean_zt = self.inf_mean(enc_output)
        logvar_zt = self.inf_logvar(enc_output)
        return mean_zt, logvar_zt

    def recurrence(self, feature_xt, feature_zt, h_t, c_t):
        rnn_input = torch.cat((feature_xt, feature_zt), -1)
        _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
        return h_tp1, c_tp1

    def forward(self, x):
        # Input is an image so reduce down to [batch_size, flattened_dim, seq_len]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(1, 0, 2)
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # create variable holder and send to GPU if needed
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.args.gpus[0])
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.args.gpus[0])
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.args.gpus[0])
        self.z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.args.gpus[0])
        h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.args.gpus[0])
        z_t = torch.zeros(batch_size, self.z_dim).to(self.args.gpus[0])
        h_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.args.gpus[0])
        c_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.args.gpus[0])

        # main part
        feature_x_obs = self.feature_extractor_x(x)
        for t in range(seq_len):
            feature_xt = feature_x_obs[t, :, :].unsqueeze(0)
            h_t_last = h_t.view(self.num_RNN, 1, batch_size, self.dim_RNN)[-1, :, :, :]
            mean_zt, logvar_zt = self.inference(feature_xt, h_t_last)
            z_t = self.reparameterization(mean_zt, logvar_zt)
            feature_zt = self.feature_extractor_z(z_t)
            y_t = self.generation_x(feature_zt, h_t_last)
            self.z_mean[t, :, :] = mean_zt
            self.z_logvar[t, :, :] = logvar_zt
            self.z[t, :, :] = torch.squeeze(z_t)
            y[t, :, :] = torch.squeeze(y_t)
            h[t, :, :] = torch.squeeze(h_t_last)
            h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t)  # recurrence for t+1

        self.z_mean_p, self.z_logvar_p = self.generation_z(h)

        # Reshape and permute reconstructions + embeddings back to useable shapes
        y = y.permute(1, 0, 2).reshape([batch_size, seq_len, self.args.dim, self.args.dim])
        embeddings = self.z.permute(1, 0, 2)
        return y, embeddings

    def model_specific_loss(self, x, x_rec, zts, train=True):
        """ KL term between the parameter distribution w and a normal prior"""
        # Reshape to [BS, SL, LatentDim]
        z_mus = self.z_mean.permute(1, 0, 2).reshape([x.shape[0], -1])
        z_logvar = self.z_logvar.permute(1, 0, 2).reshape([x.shape[0], -1])

        z_mus_prior = self.z_mean_p.permute(1, 0, 2).reshape([x.shape[0], -1])
        z_logvar_prior = self.z_logvar_p.permute(1, 0, 2).reshape([x.shape[0], -1])

        q = Normal(z_mus, torch.exp(0.5 * z_logvar))
        N = Normal(z_mus_prior, torch.exp(0.5 * z_logvar_prior))
        return kl(q, N).sum([-1]).mean()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Add the TRS regularization weight """
        parser = parent_parser.add_argument_group("NODE_TR")
        parser.add_argument('--trs_beta', type=float, default=100, help='multiplier for encoder kl terms in loss')
        return parent_parser
