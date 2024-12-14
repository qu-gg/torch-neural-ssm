"""
@file utils.py

Utility functions across files
"""
import math
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler


def flatten_cfg(cfg: DictConfig):
    """ Utility function to flatten the primary submodules of a Hydra config """
    # Disable struct flag on the config
    OmegaConf.set_struct(cfg, False)

    # Loop through each item, merging with the main cfg if its another DictConfig
    for key, value in cfg.copy().items():
        if isinstance(value, DictConfig):            
            cfg.merge_with(cfg.pop(key))

    # Do it a second time for nested cfgs
    for key, value in cfg.copy().items():
            if isinstance(value, DictConfig):            
                cfg.merge_with(cfg.pop(key))

    print(cfg)
    return cfg


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    ## Group A Models
    if name == "vrnn":
        from models.group_a.VRNN import VRNN
        return VRNN

    if name == "dkf":
        from models.group_a.DKF import DKF
        return DKF
    
    ## Group B1 Models
    if name == "kvae":
        from models.group_b1.KVAE import KVAE
        return KVAE
    
    ## Group B2 Models
    if name == "node":
        from models.group_b2.NeuralODE import NeuralODE
        return NeuralODE

    if name == "rgnres":
        from models.group_b2.RGNRes import RGNRes
        return RGNRes
    
    # Given no correct model type, raise error
    raise NotImplementedError("Model type {} not implemented.".format(name))


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None


def determine_annealing_factor(n_updates, min_anneal_factor=0.0, anneal_update=10000):
    """
    Handles annealing the KL restriction over a number of update steps to slowly introduce the regularization
    to ensure a strong initial fit has been set
    :param min_anneal_factor: minimum
    :param anneal_update: over how long of updates to apply the annealing factor
    :param epoch: current epoch number
    :param n_batch: number of total batches within an epoch
    :param batch_idx: current batch idx within the epoch
    :return: weight of the kl annealing factor for the loss term
    """
    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, warmup_steps=350, decay=1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(optimizer, last_epoch, verbose)

        # Decay attributes
        self.decay = decay
        self.initial_lrs = self.base_lrs

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0

    def get_lr(self):
        return [
            (self.current_steps / self.warmup_steps) *
            (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print("multiplying base_lrs by {:.4f}".format(self.decay))
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
