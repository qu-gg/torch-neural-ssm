"""
@file KVAE.py

Holds the model for the Kalman Variational Auto-encoder, source code modified from
@url{https://github.com/XiaoyuBIE1994/DVAE/blob/master/dvae/model/kvae.py}
"""
import torch
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from models.CommonDynamics import LatentDynamicsModel
from torch.distributions.multivariate_normal import MultivariateNormal


class FakeEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def kl_z_term(self):
        return torch.Tensor([0.]).to(self.cfg.devices[0])


class KVAE(LatentDynamicsModel):
    def __init__(self, cfg):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(cfg)
        self.encoder = FakeEncoder(cfg)

        ## General parameters
        self.x_dim = self.cfg.dim**2
        self.y_dim = self.cfg.dim**2
        self.a_dim = self.cfg.latent_dim * 2
        self.z_dim = self.cfg.latent_dim
        self.u_dim = self.cfg.latent_dim * 2
        self.dropout_p = 0.2
        self.activation = nn.ReLU()

        # VAE
        self.dense_x_a = [512, 256]
        self.dense_a_x = [512, 256]

        # LGSSM
        self.init_kf_mat = 0.05
        self.noise_transition = 0.08
        self.noise_emission = 0.03
        self.init_cov = 20

        # Dynamics params (alpha)
        self.K = 4
        self.dim_RNN_alpha = 128
        self.num_RNN_alpha = 2

        self.build()

    def build(self):
        #############
        #### VAE ####
        #############
        # 1. Inference of a_t
        dic_layers = OrderedDict()
        if len(self.dense_x_a) == 0:
            dim_x_a = self.x_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_x_a = self.dense_x_a[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(self.x_dim, self.dense_x_a[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(self.dense_x_a[n - 1], self.dense_x_a[n])
                dic_layers['activation' + str(n)] = self.activation
                dic_layers['dropout' + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_a = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_a, self.a_dim)
        self.inf_logvar = nn.Linear(dim_x_a, self.a_dim)

        # 2. Generation of x_t
        dic_layers = OrderedDict()
        if len(self.dense_a_x) == 0:
            dim_a_x = self.a_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_a_x = self.dense_a_x[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(self.a_dim, self.dense_a_x[n])
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(self.dense_a_x[n - 1], self.dense_a_x[n])
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_a_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_a_x, self.x_dim)

        ###############
        #### LGSSM ####
        ###############
        # Initializers for LGSSM variables, torch.tensor(), enforce torch.float32 type
        # A is an identity matrix
        # B and C are randomly sampled from a Gaussian
        # Q and R are isotroipic covariance matrices
        # z = Az + Bu
        # a = Cz
        self.A = torch.tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), dtype=torch.float32,
                              requires_grad=True, device=self.cfg.devices[0])  # (K, z_dim. z_dim,)
        self.B = torch.tensor(
            np.array([self.init_kf_mat * np.random.randn(self.z_dim, self.u_dim) for _ in range(self.K)]),
            dtype=torch.float32, requires_grad=True, device=self.cfg.devices[0])  # (K, z_dim, u_dim)
        self.C = torch.tensor(
            np.array([self.init_kf_mat * np.random.randn(self.a_dim, self.z_dim) for _ in range(self.K)]),
            dtype=torch.float32, requires_grad=True, device=self.cfg.devices[0])  # (K, a_dim, z_dim)
        self.Q = self.noise_transition * torch.eye(self.z_dim).to(self.cfg.devices[0])  # (z_dim, z_dim)
        self.R = self.noise_emission * torch.eye(self.a_dim).to(self.cfg.devices[0])  # (a_dim, a_dim)
        self._I = torch.eye(self.z_dim).to(self.cfg.devices[0])  # (z_dim, z_dim)

        ###############
        #### Alpha ####
        ###############
        self.a_init = torch.zeros((1, self.a_dim), requires_grad=True, device=self.cfg.devices[0])  # (bs, a_dim)
        self.rnn_alpha = nn.LSTM(self.a_dim, self.dim_RNN_alpha, self.num_RNN_alpha, bidirectional=False)
        self.mlp_alpha = nn.Sequential(nn.Linear(self.dim_RNN_alpha, self.K), nn.Softmax(dim=-1))

        ############################
        #### Scheduler Training ####
        ############################
        self.A = nn.Parameter(self.A)
        self.B = nn.Parameter(self.B)
        self.C = nn.Parameter(self.C)
        self.a_init = nn.Parameter(self.a_init)
        kf_params = [self.A, self.B, self.C, self.a_init]

        self.iter_kf = (i for i in kf_params)
        self.iter_vae = self.concat_iter(self.mlp_x_a.parameters(),
                                         self.inf_mean.parameters(),
                                         self.inf_logvar.parameters(),
                                         self.mlp_a_x.parameters(),
                                         self.gen_logvar.parameters())
        self.iter_alpha = self.concat_iter(self.rnn_alpha.parameters(),
                                           self.mlp_alpha.parameters())
        self.iter_vae_kf = self.concat_iter(self.iter_vae, self.iter_kf)
        self.iter_all = self.concat_iter(self.iter_kf, self.iter_vae, self.iter_alpha)

    def concat_iter(self, *iter_list):
        for i in iter_list:
            yield from i

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def inference(self, x):
        x_a = self.mlp_x_a(x)
        a_mean = self.inf_mean(x_a)
        a_logvar = self.inf_logvar(x_a)
        a = self.reparameterization(a_mean, a_logvar)
        return a, a_mean, a_logvar

    def get_alpha(self, a_tm1):
        """
        Dynamics parameter network alpha for mixing transitions in a SSM
        Unlike original code, we only propose RNN here
        """
        alpha, _ = self.rnn_alpha(a_tm1)  # (seq_len, bs, dim_alpha)
        alpha = self.mlp_alpha(alpha)  # (seq_len, bs, K), softmax on K dimension
        return alpha

    def generation_x(self, a):
        a_x = self.mlp_a_x(a)
        log_y = self.gen_logvar(a_x)
        y = torch.exp(log_y)
        return y

    def kf_smoother(self, a, u, K, A, B, C, R, Q, optimal_gain=False, alpha_sq=1):
        """"
        Kalman Smoother, refer to Murphy's book (MLAPP), section 18.3
        Difference from KVAE source code:
            - no imputation
            - only RNN for the calculation of alpha
            - different notations (rather than using same notations as Murphy's book ,we use notation from model KVAE)
            # z_t = A_t * z_tm1 + B_t * u_t
            #a_t = C_t * z_t
        Input:
            - a, (seq_len, bs, a_dim)
            - u, (seq_len, bs, u_dim)
            - alpha, (seq_len, bs, alpha_dim)
            - K, real number
            - A, (K, z_dim, z_dim)
            - B, (K, z_dim, u_dim)
            - C, (K, a_dim, z_dim)
            - R, (z_dim, z_dim)
            - Q , (a_dim, a_dim)
        """
        # Initialization
        seq_len = a.shape[0]
        batch_size = a.shape[1]
        self.mu = torch.zeros((batch_size, self.z_dim)).to(self.device)  # (bs, z_dim), z_0
        self.Sigma = self.init_cov * torch.eye(self.z_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)  # (bs, z_dim, z_dim), Sigma_0
        mu_pred = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim)
        mu_filter = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim)
        mu_smooth = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim)
        Sigma_pred = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim, z_dim)
        Sigma_filter = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim, z_dim)
        Sigma_smooth = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)  # (seq_len, bs, z_dim, z_dim)

        # Calculate alpha, initial observation a_init is assumed to be zero and can be learned
        a_init_expand = self.a_init.unsqueeze(1).repeat(1, batch_size, 1)  # (1, bs, a_dim)
        a_tm1 = torch.cat([a_init_expand, a[:-1, :, :]], 0)  # (seq_len, bs, a_dim)
        alpha = self.get_alpha(a_tm1)  # (seq_len, bs, K)

        # Calculate the mixture of A, B and C
        A_flatten = A.view(K, self.z_dim * self.z_dim)  # (K, z_dim*z_dim)
        B_flatten = B.view(K, self.z_dim * self.u_dim)  # (K, z_dim*u_dim)
        C_flatten = C.view(K, self.a_dim * self.z_dim)  # (K, a_dim*z_dim)
        A_mix = alpha.matmul(A_flatten).view(seq_len, batch_size, self.z_dim, self.z_dim)
        B_mix = alpha.matmul(B_flatten).view(seq_len, batch_size, self.z_dim, self.u_dim)
        C_mix = alpha.matmul(C_flatten).view(seq_len, batch_size, self.a_dim, self.z_dim)

        # Forward filter
        for t in range(seq_len):
            # Mixture of A, B and C
            A_t = A_mix[t]  # (bs, z_dim. z_dim)
            B_t = B_mix[t]  # (bs, z_dim, u_dim)
            C_t = C_mix[t]  # (bs, a_dim, z_dim)

            if t == 0:
                mu_t_pred = self.mu.unsqueeze(-1)  # (bs, z_dim, 1)
                Sigma_t_pred = self.Sigma
            else:
                u_t = u[t, :, :]  # (bs, u_dim)
                mu_t_pred = A_t.bmm(mu_t) + B_t.bmm(u_t.unsqueeze(-1))  # (bs, z_dim, 1), z_{t|t-1}
                Sigma_t_pred = alpha_sq * A_t.bmm(Sigma_t).bmm(
                    A_t.transpose(1, 2)) + self.Q  # (bs, z_dim, z_dim), Sigma_{t|t-1}
                # alpha_sq (>=1) is fading memory control, which indicates how much you want to forgert past measurements, see more infos in 'FilterPy' library

            # Residual
            a_pred = C_t.bmm(mu_t_pred)  # (bs, a_dim, z_dim) x (bs, z_dim, 1)
            res_t = a[t, :, :].unsqueeze(-1) - a_pred  # (bs, a_dim, 1)

            # Kalman gain
            S_t = C_t.bmm(Sigma_t_pred).bmm(C_t.transpose(1, 2)) + self.R  # (bs, a_dim, a_dim)
            S_t_inv = S_t.inverse()
            K_t = Sigma_t_pred.bmm(C_t.transpose(1, 2)).bmm(S_t_inv)  # (bs, z_dim, a_dim)

            # Update
            mu_t = mu_t_pred + K_t.bmm(res_t)  # (bs, z_dim, 1)
            I_KC = self._I - K_t.bmm(C_t)  # (bs, z_dim, z_dim)
            if optimal_gain:
                Sigma_t = I_KC.bmm(Sigma_t_pred)  # (bs, z_dim, z_dim), only valid with optimal Kalman gain
            else:
                Sigma_t = I_KC.bmm(Sigma_t_pred).bmm(I_KC.transpose(1, 2)) + K_t.matmul(self.R).matmul(
                    K_t.transpose(1, 2))  # (bs, z_dim, z_dim), general case

            # Save cache
            mu_pred[t] = mu_t_pred.view(batch_size, self.z_dim)
            Sigma_pred[t] = Sigma_t_pred
            Sigma_filter[t] = Sigma_t

        # Add the final state from filter to the smoother as initialization
        mu_smooth[-1] = mu_filter[-1]
        Sigma_smooth[-1] = Sigma_filter[-1]

        # Backward smooth, reverse loop from pernultimate state
        for t in range(seq_len - 2, -1, -1):
            # Backward Kalman gain
            J_t = Sigma_filter[t].bmm(A_mix[t + 1].transpose(1, 2)).bmm(
                Sigma_pred[t + 1].inverse())  # (bs, z_dim, z_dim)

            # Backward smoothing
            dif_mu_tp1 = (mu_smooth[t + 1] - mu_filter[t + 1]).unsqueeze(-1)  # (bs, z_dim, 1)
            mu_smooth[t] = mu_filter[t] + J_t.matmul(dif_mu_tp1).view(batch_size, self.z_dim)  # (bs, z_dim)
            dif_Sigma_tp1 = Sigma_smooth[t + 1] - Sigma_pred[t + 1]  # (bs, z_dim, z_dim)
            Sigma_smooth[t] = Sigma_filter[t] + J_t.bmm(dif_Sigma_tp1).bmm(J_t.transpose(1, 2))  # (bs, z_dim, z_dim)

        # Generate a from smoothing z
        a_gen = C_mix.matmul(mu_smooth.unsqueeze(-1)).view(seq_len, batch_size, self.a_dim)  # (seq_len, bs, a_dim)
        return a_gen, mu_smooth, Sigma_smooth, A_mix, B_mix, C_mix

    def forward(self, x, generation_len):
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)

        # Input is an image so reduce down to [batch_size, flattened_dim, seq_len]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(1, 0, 2)
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        self.a, self.a_mean, self.a_logvar = self.inference(x)
        u_0 = torch.zeros(1, batch_size, self.u_dim).to(self.cfg.devices[0])
        self.u = torch.cat((u_0, self.a[:-1]), 0)
        a_gen, self.mu_smooth, self.Sigma_smooth, self.A_mix, self.B_mix, self.C_mix = self.kf_smoother(
            self.a, self.u, self.K, self.A, self.B, self.C, self.R, self.Q
        )
        self.y = self.generation_x(a_gen)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        self.y = self.y.permute(1, 0, 2)

        # Reshape y to image dimensions
        self.y = self.y.reshape([batch_size, seq_len, self.cfg.dim, self.cfg.dim])
        return self.y, torch.zeros([batch_size, seq_len, self.cfg.latent_dim])

    def model_specific_loss(self, x, x_rec, train=True):
        # batch_size, seq_len = x.shape[0], x.shape[1]
        # Input is an image so reduce down to [batch_size, flattened_dim, seq_len]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(1, 0, 2)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # log q_{\phi}(a_hat | x), Gaussian
        log_qa_given_x = - 0.5 * self.a_logvar - torch.pow(self.a - self.a_mean, 2) / (2 * torch.exp(self.a_logvar))

        # log p_{\gamma}(a_tilde, z_tilde | u) < in sub-comment, 'tilde' is hidden for simplification >
        # >>> log p(z_t | z_tm1, u_t), transition
        Sigma_smooth_stable = self.Sigma_smooth
        L = torch.linalg.cholesky(Sigma_smooth_stable)
        mvn_smooth = MultivariateNormal(self.mu_smooth, scale_tril=L)

        
        # mvn_smooth = MultivariateNormal(self.mu_smooth, self.Sigma_smooth)
        z_smooth = mvn_smooth.sample()  # # (seq_len, bs, z_dim)
        Az_tm1 = self.A_mix[:-1].matmul(z_smooth[:-1].unsqueeze(-1)).view(seq_len - 1, batch_size, -1)  # (seq_len, bs, z_dim)
        Bu_t = self.B_mix[:-1].matmul(self.u[:-1].unsqueeze(-1)).view(seq_len - 1, batch_size, -1)  # (seq_len, bs, z_dim)
        mu_t_transition = Az_tm1 + Bu_t
        z_t_transition = z_smooth[1:]
        mvn_transition = MultivariateNormal(z_t_transition, self.Q)
        log_prob_transition = mvn_transition.log_prob(mu_t_transition)

        # >>> log p(z_0 | z_init), init state
        z_0 = z_smooth[0]
        mvn_0 = MultivariateNormal(self.mu, self.Sigma)
        log_prob_0 = mvn_0.log_prob(z_0)

        # >>> log p(a_t | z_t), emission
        Cz_t = self.C_mix.matmul(z_smooth.unsqueeze(-1)).view(seq_len, batch_size, self.a_dim)
        mvn_emission = MultivariateNormal(Cz_t, self.R)
        log_prob_emission = mvn_emission.log_prob(self.a)

        # >>> log p_{\gamma}(a_tilde, z_tilde | u)
        log_paz_given_u = torch.cat([log_prob_transition, log_prob_0.unsqueeze(0)], 0) + log_prob_emission

        # log p_{\gamma}(z_tilde | a_tilde, u)
        # >>> log p(z_t | a, u)
        log_pz_given_au = mvn_smooth.log_prob(z_smooth)

        # Normalization
        log_qa_given_x = torch.sum(log_qa_given_x) / (batch_size * seq_len)
        log_paz_given_u = torch.sum(log_paz_given_u) / (batch_size * seq_len)
        log_pz_given_au = torch.sum(log_pz_given_au) / (batch_size * seq_len)

        # Loss
        loss_vae = log_qa_given_x
        loss_lgssm = - log_paz_given_u + log_pz_given_au
        loss_tot = loss_vae + loss_lgssm
        return loss_vae + loss_lgssm + loss_tot
