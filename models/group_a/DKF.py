"""
@file DKF.py

Holds the model for the Deep Kalman Filter baseline, based off of the code from
@url{https://github.com/john-x-jiang/meta_ssm/blob/main/model/model.py}
"""
import torch
import torch.nn as nn
import torch.nn.init as weight_init

from models.CommonDynamics import LatentDynamicsModel
from torch.distributions import Normal, kl_divergence as kl


def reverse_sequence(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py
    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )
    Returns
    -------
    x_reverse: tensor (b, T_max, input_dim)
        The input x in reversed order w.r.t. time-axis
    """
    x_reverse = torch.zeros_like(x)
    for b in range(x.size(0)):
        t = seq_lengths[b]
        time_slice = torch.arange(t - 1, -1, -1, device=x.device)
        reverse_seq = torch.index_select(x[b, :, :], 0, time_slice)
        x_reverse[b, 0:t, :] = reverse_seq

    return x_reverse


class RnnEncoder(nn.Module):
    """
    RNN encoder that outputs hidden states h_t using x_{t:T}
    Parameters
    ----------
    input_dim: int
        Dim. of inputs
    num_hidden: int
        Dim. of RNN hidden states
    n_layer: int
        Number of layers of RNN
    drop_rate: float [0.0, 1.0]
        RNN dropout rate between layers
    bd: bool
        Use bi-directional RNN or not
    Returns
    -------
    h_rnn: tensor (b, T_max, num_hidden * n_direction)
        RNN hidden states at every time-step
    """
    def __init__(self, cfg, input_dim, num_hidden, n_layer=1, drop_rate=0.0, bd=False, nonlin='relu',
                 rnn_type='rnn', orthogonal_init=False, reverse_input=True):
        super().__init__()
        self.n_direction = 1 if not bd else 2
        self.cfg = cfg
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.bd = bd
        self.nonlin = nonlin
        self.reverse_input = reverse_input

        if not isinstance(rnn_type, str):
            raise ValueError("`rnn_type` should be type str.")
        self.rnn_type = rnn_type
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=num_hidden, nonlinearity=nonlin,
                              batch_first=True, bidirectional=bd, num_layers=n_layer, dropout=drop_rate)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=num_hidden,
                              batch_first=True, bidirectional=bd, num_layers=n_layer, dropout=drop_rate)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=num_hidden, batch_first=True,
                               bidirectional=bd, num_layers=n_layer, dropout=drop_rate)
        else:
            raise ValueError("`rnn_type` must instead be ['rnn', 'gru', 'lstm'] %s" % rnn_type)

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def calculate_effect_dim(self):
        return self.num_hidden * self.n_direction

    def init_hidden(self, trainable=True):
        if self.rnn_type == 'lstm':
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.num_hidden), requires_grad=trainable)
            c0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.num_hidden), requires_grad=trainable)
            return h0, c0
        else:
            h0 = nn.Parameter(torch.zeros(self.n_layer * self.n_direction, 1, self.num_hidden), requires_grad=trainable)
            return h0

    def kl_z_term(self):
        """
        KL Z term, KL[q(z0|X) || N(0,1)]
        :return: mean klz across batch
        """
        return torch.Tensor([0]).to(self.cfg.devices[0])

    def forward(self, x):
        """
        x: pytorch packed object
            input packed data; this can be obtained from
            `util.get_mini_batch()`
        h0: tensor (n_layer * n_direction, b, num_hidden)
        """
        B, T, _ = x.shape
        seq_lengths = T * torch.ones(B).int().to(self.cfg.devices[0])
        h_rnn, _ = self.rnn(x)
        if self.reverse_input:
            h_rnn = reverse_sequence(h_rnn, seq_lengths)
        return h_rnn


class Transition_Recurrent(nn.Module):
    """
    Parameterize the diagonal Gaussian latent transition probability
    `p(z_t | z_{t-1})`
    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    transition_dim: int
        Dim. of transition hidden units
     gated: bool
        Use the gated mechanism to consider both linearity and non-linearity
    identity_init: bool
        Initialize the linearity transform as an identity matrix;
        ignored if `gated == False`
    clip: bool
        clip the value for numerical issues
    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the Gaussian
    logvar: tensor (b, z_dim)
        Log-variance that parameterizes the Gaussian
    """

    def __init__(self, z_dim, transition_dim, identity_init=True, domain=False, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.identity_init = identity_init
        self.domain = domain
        self.stochastic = stochastic

        if domain:
            # compute the gain (gate) of non-linearity
            self.lin1 = nn.Linear(z_dim * 2, transition_dim * 2)
            self.lin2 = nn.Linear(transition_dim * 2, z_dim)
            # compute the proposed mean
            self.lin3 = nn.Linear(z_dim * 2, transition_dim * 2)
            self.lin4 = nn.Linear(transition_dim * 2, z_dim)
            # linearity
            self.lin0 = nn.Linear(z_dim * 2, z_dim)
        else:
            # compute the gain (gate) of non-linearity
            self.lin1 = nn.Linear(z_dim, transition_dim)
            self.lin2 = nn.Linear(transition_dim, z_dim)
            # compute the proposed mean
            self.lin3 = nn.Linear(z_dim, transition_dim)
            self.lin4 = nn.Linear(transition_dim, z_dim)
            self.lin0 = nn.Linear(z_dim, z_dim)

        # compute the linearity part
        self.lin_n = nn.Linear(z_dim, z_dim)

        if identity_init:
            self.lin_n.weight.data = torch.eye(z_dim)
            self.lin_n.bias.data = torch.zeros(z_dim)

        # compute the variation
        self.lin_v = nn.Linear(z_dim, z_dim)
        # var activation
        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

        self.act_weight = nn.Sigmoid()
        self.act = nn.ELU(inplace=True)

    def init_z_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
               nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, z_t_1, z_c=None):
        if self.domain:
            z_combine = torch.cat((z_t_1, z_c), dim=1)
            _g_t = self.act(self.lin1(z_combine))
            g_t = self.act_weight(self.lin2(_g_t))
            _h_t = self.act(self.lin3(z_combine))
            h_t = self.act(self.lin4(_h_t))
            _mu = self.lin0(z_combine)
            mu = (1 - g_t) * self.lin_n(_mu) + g_t * h_t
            mu = mu + _mu
        else:
            _g_t = self.act(self.lin1(z_t_1))
            g_t = self.act_weight(self.lin2(_g_t))
            _h_t = self.act(self.lin3(z_t_1))
            h_t = self.act(self.lin4(_h_t))
            mu = (1 - g_t) * self.lin_n(z_t_1) + g_t * h_t
            _mu = self.lin0(z_t_1)
            mu = mu + _mu

        if self.stochastic:
            _var = self.lin_v(h_t)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class Correction(nn.Module):
    """
    Parameterize variational distribution `q(z_t | z_{t-1}, x_{t:T})`
    a diagonal Gaussian distribution
    Parameters
    ----------
    z_dim: int
        Dim. of latent variables
    num_hidden: int
        Dim. of RNN hidden states
    clip: bool
        clip the value for numerical issues
    Returns
    -------
    mu: tensor (b, z_dim)
        Mean that parameterizes the variational Gaussian distribution
    logvar: tensor (b, z_dim)
        Log-var that parameterizes the variational Gaussian distribution
    """
    def __init__(self, z_dim, num_hidden, stochastic=True):
        super().__init__()
        self.z_dim = z_dim
        self.num_hidden = num_hidden
        self.stochastic = stochastic

        self.lin1 = nn.Linear(z_dim, num_hidden)
        self.act = nn.Tanh()

        self.lin2 = nn.Linear(num_hidden, z_dim)
        self.lin_v = nn.Linear(num_hidden, z_dim)

        # self.act_var = nn.Softplus()
        self.act_var = nn.Tanh()

    def init_z_q_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)

    def forward(self, h_rnn, z_t_1=None, rnn_bidirection=False):
        """
        z_t_1: tensor (b, z_dim)
        h_rnn: tensor (b, num_hidden)
        """
        assert z_t_1 is not None
        h_comb_ = self.act(self.lin1(z_t_1))
        if rnn_bidirection:
            h_comb = (1.0 / 3) * (h_comb_ + h_rnn[:, :self.num_hidden] + h_rnn[:, self.num_hidden:])
        else:
            h_comb = 0.5 * (h_comb_ + h_rnn)
        mu = self.lin2(h_comb)

        if self.stochastic:
            _var = self.lin_v(h_comb)
            _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class DKF(LatentDynamicsModel):
    def __init__(self, cfg):
        """ Deep Kalman Filter model """
        super().__init__(cfg)

        # observation
        self.embedding = nn.Sequential(
            nn.Linear(self.cfg.dim**2, self.cfg.dim**2 // 4),
            nn.ReLU(),
            # nn.Linear(self.cfg.dim**2 // 4, self.cfg.dim**2 // 8),
            # nn.ReLU(),
            nn.Linear(self.cfg.dim**2 // 4, self.cfg.num_hidden),
            nn.ReLU()
        )
        self.encoder = RnnEncoder(cfg, self.cfg.num_hidden, self.cfg.num_hidden, n_layer=self.cfg.rnn_layers, drop_rate=0.0,
                                  bd=self.cfg.bidirectional, nonlin='relu', rnn_type=self.cfg.rnn_type, reverse_input=False)

        # generative model
        self.transition = Transition_Recurrent(z_dim=self.cfg.latent_dim, transition_dim=self.cfg.transition_dim)
        self.estimation = Correction(z_dim=self.cfg.latent_dim, num_hidden=self.cfg.num_hidden, stochastic=True)

        # initialize hidden states
        self.mu_p_0, self.var_p_0 = self.transition.init_z_0(trainable=False)
        self.z_q_0 = self.estimation.init_z_q_0(trainable=False)

        # hold p and q distribution parameters for KL term
        self.mu_qs = None
        self.var_qs = None
        self.mu_ps = None
        self.var_ps = None
        
        # placeholder for z_amort
        self.z_amort = None

    def reparameterization(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_dynamics(self, T, x_rnn):
        batch_size = x_rnn.shape[0]

        if T > self.z_amort:
            T_final = T
        else:
            T_final = self.z_amort

        z_ = torch.zeros([batch_size, T_final, self.cfg.latent_dim]).to(self.cfg.devices[0])
        if not self.cfg.use_q_forecast:
            mu_ps = torch.zeros([batch_size, self.z_amort, self.cfg.latent_dim]).to(self.cfg.devices[0])
            var_ps = torch.zeros([batch_size, self.z_amort, self.cfg.latent_dim]).to(self.cfg.devices[0])
            mu_qs = torch.zeros([batch_size, self.z_amort, self.cfg.latent_dim]).to(self.cfg.devices[0])
            var_qs = torch.zeros([batch_size, self.z_amort, self.cfg.latent_dim]).to(self.cfg.devices[0])
        else:
            mu_ps = torch.zeros([batch_size, T_final, self.cfg.latent_dim]).to(self.cfg.devices[0])
            var_ps = torch.zeros([batch_size, T_final, self.cfg.latent_dim]).to(self.cfg.devices[0])
            mu_qs = torch.zeros([batch_size, T_final, self.cfg.latent_dim]).to(self.cfg.devices[0])
            var_qs = torch.zeros([batch_size, T_final, self.cfg.latent_dim]).to(self.cfg.devices[0])
            x_qs = torch.zeros([batch_size, T_final, self.cfg.dim, self.cfg.dim]).to(self.cfg.devices[0])

        z_q_0 = self.z_q_0.expand(batch_size, self.cfg.latent_dim)  # q(z_0)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.cfg.latent_dim)
        var_p_0 = self.var_p_0.expand(batch_size, 1, self.cfg.latent_dim)
        z_prev = z_q_0
        z_[:, 0, :] = z_prev

        for t in range(self.z_amort):
            # zt = self.transition(z_prev)
            mu_q, var_q = self.estimation(x_rnn[:, t, :], z_prev, rnn_bidirection=self.cfg.bidirectional)
            zt_q = self.reparameterization(mu_q, var_q)
            z_prev = zt_q

            # p(z_{t+1} | z_t)
            mu_p, var_p = self.transition(z_prev)
            zt_p = self.reparameterization(mu_p, var_p)

            z_[:, t, :] = zt_q
            mu_qs[:, t, :] = mu_q
            var_qs[:, t, :] = var_q
            mu_ps[:, t, :] = mu_p
            var_ps[:, t, :] = var_p

        if T > self.z_amort:
            for t in range(self.z_amort, T):
                if self.cfg.use_q_forecast:
                    y = self.decoder(z_prev.unsqueeze(1))
                    x_qs[:,t,:,:] = y.squeeze()
                    y = self.encoder(self.embedding(y.view(y.size(0), 1, -1)))
                    mu_q, var_q = self.estimation(y[:,0,:], z_prev, rnn_bidirection=self.cfg.bidirectional)
                    zt_q = self.reparameterization(mu_q, var_q)
                    z_prev = zt_q
                    
                    mu_p, var_p = self.transition(z_prev)
                    zt_p = self.reparameterization(mu_p, var_p)

                    z_[:, t, :] = zt_q
                    mu_qs[:, t, :] = mu_q
                    var_qs[:, t, :] = var_q
                    mu_ps[:, t, :] = mu_p
                    var_ps[:, t, :] = var_p
                else:
                    # p(z_{t+1} | z_t)
                    mu_p, var_p = self.transition(z_prev)
                    zt_p = self.reparameterization(mu_p, var_p)
                    z_[:, t, :] = zt_p
                    z_prev = zt_p

        mu_ps = torch.cat([mu_p_0, mu_ps[:, :-1, :]], dim=1)
        var_ps = torch.cat([var_p_0, var_ps[:, :-1, :]], dim=1)

        self.mu_qs, self.var_qs = mu_qs, var_qs
        self.mu_ps, self.var_ps = mu_ps, var_ps
        if self.cfg.use_q_forecast:
            return z_, mu_qs, var_qs, mu_ps, var_ps, x_qs
        else:    
            return z_, mu_qs, var_qs, mu_ps, var_ps

    def forward(self, x, generation_len):
        if self.trainer.training:
            self.z_amort = self.cfg.z_amort_train
        else:
            self.z_amort = self.cfg.z_amort_test
            
        batch_size = x.size(0)

        x = x.view(batch_size, generation_len, -1)
        x = self.embedding(x)
        x_rnn = self.encoder(x)

        if not self.cfg.use_q_forecast:
            z_, mu_qs, var_qs, mu_ps, var_ps = self.latent_dynamics(generation_len, x_rnn)
            x_ = self.decoder(z_)
        else:
            z_, mu_qs, var_qs, mu_ps, var_ps, x_ = self.latent_dynamics(generation_len, x_rnn)
            x_[:,:self.z_amort,:,:] = self.decoder(z_[:,:self.z_amort,:])
        return x_, z_

    def model_specific_loss(self, *args, train=True):
        """ KL term between the p and q distributions (reconstruction and estimation) """
        q = Normal(self.mu_qs, torch.exp(0.5 * self.var_qs))
        p = Normal(self.mu_ps, torch.exp(0.5 * self.var_ps))
        return kl(q, p).sum([-1]).mean()
