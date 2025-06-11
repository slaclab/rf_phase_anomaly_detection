"""This is a simplified version of the prediction function from the original code.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_train.py
"""

import os
from operator import itemgetter
from typing import Any, List, Dict

import torch
import yaml

from lume_model.models.torch_module import TorchModule


ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class Predict:
    """Predict class for making predictions using LUME-models."""

    def __init__(self, configs=None, networks=None):
        """Initialize the Predict class.
        Args:
            configs (dict, optional): Configuration dictionary for the prediction.
            networks (list, optional): List of TorchModule instances representing the models.
        """
        self.configs = load_configs() or configs
        self.networks = load_models() or networks

    def predict(self, batch: List[torch.Tensor]) -> bool:
        """Make predictions using the loaded models and provided a single batch of data.
        Args:
            batch (list): List of input data for the models, should be torch tensors and have a length of 2. The first
            element should be the RF data with a shape of (D, N), where N is the number of samples and D is the number
            of RFs (1), and the second element should be the BPM data with a shape of (D, N), where D is the number
            of BPMs (8).
        Returns:
            bool: Prediction result, True if an anomaly is detected, False otherwise.
        """
        return predict(self.configs, self.networks, batch)


def load_models():
    """Load models using LUME-model TorchModule class.

    The LUME-models should be dumped into YAML files in the models directory.
    Note that when the models are updated, new YAML files should be created along
    with the new model weights (.pt files). If the model structure/class changes, the configs.yml
    and the model class should also be updated accordingly. This function assumes that has been set up correctly,
    and that the YAML files are compatible with the TorchModule class.

    If model updates are done through MLflow for example, this function should be updated to load the models
    from MLflow instead of the YAML files, but should still return a list of TorchModule instances.

    Returns:
        List of TorchModule instances loaded from YAML files.
    """
    return [
        TorchModule(ROOTDIR + "/models/rf_module.yml"),
        TorchModule(ROOTDIR + "/models/bpm_module.yml"),
    ]


def load_configs():
    """Load configurations from a YAML file.
    The configs should have the following keys:
        - predict: dict with keys 'thres', 'exact', 'standardize', 'sigmoid', 'device'
        - network: dict with information about the networks, optional and for reference only.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """

    with open(ROOTDIR + "/configs.yml", "r") as file:
        return yaml.safe_load(file)


def predict(
    configs: Dict[str, Any], networks: List[TorchModule], batch: List[torch.Tensor]
) -> bool:
    """Make predictions using the loaded models and provided a single batch of data.

    Args:
        configs (dict): Configuration dictionary for the prediction.
        networks (list): List of TorchModule instances representing the models (length should be 2).
        batch (list): List of input data for the models, should be torch tensors and have a length of 2. The first
        element should be the RF data with a shape of (D, N), where N is the number of samples and D is the number
        of RFs (1), and the second element should be the BPM data with a shape of (D, N), where D is the number
        of BPMs (8).
    Returns:
        bool: Prediction result, True if an anomaly is detected, False otherwise.
    """
    _validate_input(configs, networks, batch)

    # Setup configs
    thres, exact, standardize, sigm, device = itemgetter(
        "thres", "exact", "standardize", "sigmoid", "device"
    )(configs["predict"])

    # Choose device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # Predict
    Y_sigm = []
    for i, net in enumerate(networks):
        X = batch[i].double().to(device)
        Y = torch.squeeze(net(X))
        if standardize:
            # Note that we do not want to standardize unless standardization was used
            # during training when calculating the loss
            Y = standardize_tensor(Y)
        if sigm:
            Y_sigm.append(torch.sigmoid(Y))
        else:
            Y_sigm.append(Y)

    S = Y_sigm[0]
    Q = Y_sigm[1]

    # Get label based on thresholds
    label = S > thres[0] and Q > thres[1]

    return bool(label)


def standardize_tensor(x):
    return (x - torch.mean(x)) / torch.std(x)


def _validate_input(configs, networks, batch):
    """Validate the input batch for the predict function."""
    if len(batch) != 2:
        raise ValueError(
            "Batch must contain exactly two elements: RF data and BPM data."
        )
    if len(networks) != 2:
        raise ValueError(
            "Networks must contain exactly two TorchModule instances: one for RF and one for BPM."
        )
    if not isinstance(networks[0], TorchModule) or not isinstance(
        networks[1], TorchModule
    ):
        raise TypeError("Networks must be instances of TorchModule.")
    if not isinstance(batch[0], torch.Tensor) or not isinstance(batch[1], torch.Tensor):
        raise TypeError("Batch elements must be torch.Tensor instances.")
    if batch[0].dim() != 2 or batch[1].dim() != 2:
        raise ValueError("Batch elements must be 2D tensors (shape: (D, N)).")
    if batch[0].shape[0] != 1 or batch[1].shape[0] != 8:
        raise ValueError(
            "RF data must have shape (1, N) and BPM data must have shape (8, N)."
        )
    if batch[0].shape[1] != batch[1].shape[1]:
        raise ValueError(
            "RF data and BPM data must have the same number of samples (N)."
        )
    if not isinstance(configs, dict):
        raise TypeError("Configs must be a dictionary.")
    if "predict" not in configs or not isinstance(configs["predict"], dict):
        raise ValueError(
            "Configs must contain a 'predict' key with a dictionary value."
        )
    if "network" in configs and not isinstance(configs["network"], list):
        raise ValueError(
            "If 'network' key is present in configs, it must be a list of two dictionaries."
        )
    if not all(
        key in configs["predict"]
        for key in ["thres", "exact", "standardize", "sigmoid", "device"]
    ):
        raise ValueError(
            "Configs['predict'] must contain 'thres', 'exact', 'standardize', 'sigmoid', and 'device' keys."
        )
