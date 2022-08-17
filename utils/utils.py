"""
@file utils.py

Utility functions across files
"""
import os
import torch.nn as nn


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    # State Estimation Models
    if name == "node_se":
        from models.state_estimation.NeuralODE_SE import NeuralODE_SE
        return NeuralODE_SE
    elif name == "lstm_se":
        from models.state_estimation.LSTM_SE import LSTM_SE
        return LSTM_SE
    elif name == "vrnn":
        from models.state_estimation.VRNN import VRNN
        return VRNN
    elif name == "dkf":
        from models.state_estimation.DKF import DKF
        return DKF

    # System Identification Models
    if name == "node_si":
        from models.system_identification.NeuralODE_SI import NeuralODE_SI
        return NeuralODE_SI
    elif name == "lstm_si":
        from models.system_identification.LSTM_SI import LSTM_SI
        return LSTM_SI
    elif name == "rgnres":
        from models.system_identification.RGNRes import RGNRes
        return RGNRes
    elif name == "rgn":
        from models.system_identification.RGN import RGN
        return RGN
    elif name == "dvbf":
        from models.system_identification.DVBF import DVBF
        return DVBF
    elif name == "kvae":
        from models.system_identification.KVAE import KVAE
        return KVAE

    # Given no correct, raise error
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
        return nn.LeakyReLU()
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

    if not os.path.exists("experiments/{}".format(model)):
        os.mkdir("experiments/{}/".format(model))

    if not os.path.exists("experiments/{}/{}".format(model, exptype)):
        os.mkdir("experiments/{}/{}".format(model, exptype))

    # Find version folder path
    exptop = 0
    for folder in os.listdir("experiments/{}/{}/".format(model, exptype)):
        try:
            num = int(folder.split("_")[-1])
            exptop = num if num > exptop else exptop
        except ValueError:
            continue

    exptop += 1
    print("Exp Top {}".format(exptop))
    return top, exptop
