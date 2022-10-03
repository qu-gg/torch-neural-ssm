"""
@file utils.py

Utility functions across files
"""
import os
import torch.nn as nn


def get_model(name, system_identification):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    #### Models

    # Neural ODE
    if name == "node":
        if system_identification is True:
            from models.system_identification.NeuralODE import NeuralODE
        else:
            from models.state_estimation.NeuralODE import NeuralODE
        return NeuralODE

    # Recurrent Generative Network (Residual)
    if name == "rgnres":
        if system_identification is True:
            from models.system_identification.RGNRes import RGNRes
        else:
            raise NotImplementedError("RGN-Res State Estimation is not yet implemented.")
        return RGNRes

    # Recurrent Generative Network
    if name == "rgn":
        if system_identification is True:
            from models.system_identification.RGN import RGN
        else:
            raise NotImplementedError("RGN State Estimation is not yet implemented.")
        return RGN

    # Long Short-Term Memory Cell
    if name == "lstm":
        if system_identification is True:
            from models.system_identification.LSTM_SI import LSTM_SI
        else:
            raise NotImplementedError("LSTM State Estimation is not yet implemented.")
        return LSTM_SI

    #### Baselines

    # Variational Recurrent Neural Network
    if name == "vrnn":
        from models.state_estimation.VRNN import VRNN
        return VRNN

    # Deep Kalman Filter
    if name == "dkf":
        from models.state_estimation.DKF import DKF
        return DKF

    # State Estimation Models
    if name == "lstm_se":
        from models.state_estimation.LSTM_SE import LSTM_SE
        return LSTM_SE

    # Deep Variational Bayes Filter
    if name == "dvbf":
        from models.system_identification.DVBF import DVBF
        return DVBF

    # Kalman Variational Auto-encoder
    if name == "kvae":
        from models.system_identification.KVAE import KVAE
        return KVAE

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


def get_exp_versions(model, exptype):
    """ Return the version number for the latest lightning log and experiment num """
    # Find version folder path
    top = 0
    for folder in os.listdir("lightning_logs/"):
        try:
            num = int(folder.split("_")[-1])
            top = num if num > top else top
        except ValueError:
            continue

    top += 1
    print("Version {}".format(top))

    # Set up paths if they don't exist
    if not os.path.exists("experiments/"):
        os.mkdir("experiments/")

    if not os.path.exists("experiments/{}".format(exptype)):
        os.mkdir("experiments/{}/".format(exptype))

    if not os.path.exists("experiments/{}/{}".format(exptype, model)):
        os.mkdir("experiments/{}/{}".format(exptype, model))

    # Find version folder path
    exptop = 0
    for folder in os.listdir("experiments/{}/{}/".format(exptype, model)):
        try:
            num = int(folder.split("_")[-1])
            exptop = num if num > exptop else exptop
        except ValueError:
            continue

    exptop += 1
    print("Exp Top {}".format(exptop))
    return top, exptop


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
