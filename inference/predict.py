"""This is a simplified version of the prediction function from the original code.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_train.py
"""

import os
import torch
import yaml

from lume_model.models.torch_module import TorchModule
from operator import itemgetter


ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class Predict:
    def __init__(self, configs=None, networks=None):
        self.configs = load_configs() or configs
        self.networks = load_models() or networks

    def predict(self, batch):
        # TODO: add validation for batch
        return predict(self.configs, self.networks, batch)


def load_models():
    # Load networks from LUME-model
    return [
        TorchModule(ROOTDIR + "/models/rf_module.yml"),
        TorchModule(ROOTDIR + "/models/bpm_module.yml"),
    ]


def load_configs():
    with open(ROOTDIR + "/configs.yml", "r") as file:
        return yaml.safe_load(file)


def predict(configs, networks, batch):
    # Setup configs
    # TODO: add a flag to use standardization
    n_pred, thres, exact = itemgetter("n_pred", "thres", "exact")(configs["predict"])
    standardize, sigm, device = itemgetter("standardize", "sigmoid", "device")(
        configs["predict"]
    )

    # Choose device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    Y_sigm = []
    for i, net in enumerate(networks):
        X = batch[i].double().to(device)
        Y = torch.squeeze(net(X))
        if standardize:
            Y = standardize_tensor(Y)
        if sigm:
            Y_sigm.append(torch.sigmoid(Y))
        else:
            Y_sigm.append(Y)

    labels = []
    Y_sigm[0] = Y_sigm[0].reshape(-1, n_pred)
    Y_sigm[1] = Y_sigm[1].reshape(-1, n_pred)

    for i in range(n_pred):
        S = Y_sigm[0][:, i]
        Q = Y_sigm[1][:, i]
        if not exact:
            flip = set_flip01_flag(S, Q, t_s=thres[0], t_b=thres[1])
            if flip:
                S = 1 - S
                Q = 1 - Q

        _labels = torch.logical_and(S > thres[0], Q > thres[1])
        labels.append(_labels)

    L = torch.vstack(labels).T.tolist()

    return L


def standardize_tensor(x):
    return (x - torch.mean(x)) / torch.std(x)


def set_flip01_flag(s_out, b_out, t_s=0.5, t_b=0.5):
    # process NN outputs to convert to anomaly predictions.
    # Main purpose is to figure out if 0 or 1 corresponds to anomalies
    # note that for equal probabilities, True and False classes are arbitrary and need to be checked against GT

    n_pts = s_out.shape[0]

    ad_pred_s = torch.zeros(n_pts, dtype=bool)
    ad_pred_b = torch.zeros(n_pts, dtype=bool)

    ad_pred_s[s_out >= t_s] = True  # predicted s anomalies
    ad_pred_b[b_out >= t_b] = True  # predicted b anomalies

    joint_pos = torch.logical_and(ad_pred_s, ad_pred_b)
    joint_neg = torch.logical_not(joint_pos)

    if torch.sum(joint_pos) > torch.sum(joint_neg):  # anomalies defined as 0
        return 1
    else:
        return 0
