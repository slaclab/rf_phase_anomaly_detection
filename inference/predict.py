"""This is a simplified version of the prediction function from the original code.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_train.py
"""

import os
from operator import itemgetter
from typing import Any, Tuple, Dict, List, Optional
import numpy.typing as npt
from numpy import number

import torch
import yaml

from lume_model.models.torch_module import TorchModule
from anom_table import set_anomaly_state, TimedBoolDict

ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class Predict:
    """Predict class for making predictions using LUME-models."""

    def __init__(
        self,
        configs: Optional[Dict[str, Any]] = None,
        networks: Optional[List[TorchModule]] = None,
        write_to_pv: bool = False,
    ) -> None:
        """
        Initialize the Predict class.

        Parameters
        ----------
        configs : dict, optional
            Configuration dictionary for the prediction.
        networks : list, optional
            List of TorchModule instances representing the models.
        write_to_pv : bool, optional
            Whether to write the prediction result to a PV. Defaults to False.
        """
        self.configs = configs if configs else load_configs()
        self.networks = networks if networks else load_models()
        self.write_to_pv = write_to_pv
        self.klystrons_list = load_klystron_configs()
        self.anom_state_dict = TimedBoolDict(self.klystrons_list, self.write_to_pv)

    def predict(
        self,
        rf_input: npt.NDArray[number],
        bpm_input: npt.NDArray[number],
        pv_name: str,
        timestamp: float,  # in nanoseconds since epoch
    ) -> bool:
        """
        Make predictions using the loaded models and provided a single batch of data.

        Parameters
        ----------
        rf_input : npt.NDArray[number]
            Numpy array of input data for the first model, with a shape of (D, N), where D is the
            number of RF stations (1) and N is the number of samples (1066).
        bpm_input : npt.NDArray[number]
            Numpy array of input data for the second model, with a shape of (D, N), where D (1066) is
            the number of BPMs (8) and N is the number of samples (1066).
        pv_name : str
            The PV name of the RF station to write the prediction result to K2EG.
        timestamp: float
            Timestamp of the prediction.

        Returns
        -------
        bool
            Prediction result, True if an anomaly is detected, False otherwise.
        """
        rf_input = torch.tensor(rf_input, dtype=torch.float64)
        bpm_input = torch.tensor(bpm_input, dtype=torch.float64)

        anomalous = predict_label(self.configs, self.networks, (rf_input, bpm_input))
        if anomalous:
            # Update anomaly state in the timed dict
            # if write to PV is enabled, it will also write to K2EG
            set_anomaly_state(self.anom_state_dict, pv_name, anomalous)
        return anomalous


def load_models() -> List[TorchModule]:
    """
    Load models using LUME-model TorchModule class.

    The LUME-models should be dumped into YAML files in the models directory.
    Note that when the models are updated, new YAML files should be created along
    with the new model weights (.pt files). If the model structure/class changes, the configs.yml
    and the model class should also be updated accordingly. This function assumes that has been set up correctly,
    and that the YAML files are compatible with the TorchModule class.

    If model updates are done through MLflow for example, this function should be updated to load the models
    from MLflow instead of the YAML files, but should still return a list of TorchModule instances.

    Returns
    -------
    list of TorchModule
        List of TorchModule instances loaded from YAML files.

    Raises
    ------
    FileNotFoundError
        If the model YAML files are not found in the expected directory.
    """
    rf_model_path = ROOTDIR + "/models/rf_module.yml"
    bpm_model_path = ROOTDIR + "/models/bpm_module.yml"
    if not os.path.exists(rf_model_path) or not os.path.exists(bpm_model_path):
        raise FileNotFoundError(
            "Model YAML files not found. Please ensure the models directory contains rf_module.yml and bpm_module.yml."
        )
    return [
        TorchModule(ROOTDIR + "/models/rf_module.yml"),
        TorchModule(ROOTDIR + "/models/bpm_module.yml"),
    ]


def load_configs() -> Dict[str, Any]:
    """
    Load configurations from a YAML file.

    The configs should have the following keys:
        - predict: dict with keys 'thres', 'exact', 'standardize', 'sigmoid', 'device'
        - network: dict with information about the networks, optional and for reference only.

    Returns
    -------
    dict
        Configuration dictionary loaded from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the configuration file 'configs.yml' is not found in the expected directory.
    """
    if not os.path.exists(ROOTDIR + "/configs.yml"):
        raise FileNotFoundError(
            "Configuration file 'configs.yml' not found in the root directory."
        )
    with open(ROOTDIR + "/configs.yml", "r") as file:
        return yaml.safe_load(file)


def load_klystron_configs() -> List:
    """
    Load klystron configurations from a YAML file.

    The configs should have the following keys:
        - klystrons: list of klystron station names.

    Returns
    -------
    List
        List of klystron names loaded from the YAML file.
    """
    with open(ROOTDIR + "/klystrons.yml", "r") as file:
        return yaml.safe_load(file)["klystrons"]


def predict_label(
    configs: Dict[str, Any],
    networks: List[TorchModule],
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> bool:
    """
    Make predictions using the loaded models and provided a single batch of data.

    Parameters
    ----------
    configs : dict
        Configuration dictionary for the prediction.
    networks : list of TorchModule
        List of TorchModule instances representing the models (length should be 2).
    batch : tuple of torch.Tensor
        Tuple of input data for the models, should be torch tensors and have a length of 2. The first
        element should be the RF data with a shape of (D, N), where D is the number of RFs (1) and N is the number
        of samples (1066), and the second element should be the BPM data with a shape of (D, N), where D is the number
        of BPMs (8) and N is the number of samples (1066).

    Returns
    -------
    bool
        Prediction result, True if an anomaly is detected, False otherwise.
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
        if standardize:
            X = standardize_tensor(X)
        Y = torch.squeeze(net(X))
        # if standardize:
        #     # Note that we do not want to standardize unless standardization was used
        #     # during training when calculating the loss
        #     Y = standardize_tensor(Y)
        if sigm:
            Y_sigm.append(torch.sigmoid(Y))
        else:
            Y_sigm.append(Y)

    S = Y_sigm[0]
    Q = Y_sigm[1]

    # Get label based on thresholds
    label = S > thres[0] and Q > thres[1]

    return bool(label)


def standardize_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Standardize the input tensor based on how the model was trained.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Standardized tensor.
    """
    if x.shape[0] == 1:
        return x - torch.median(x, dim=1, keepdim=True)[0]
    if x.shape[0] == 8:
        denom = torch.tensor([
            5.8287e-02,
            6.3393e-02,
            7.5411e-01,
            3.6302e-01,
            2.0103e07,
            2.8120e08,
            2.7878e08,
            2.7979e08,
        ]).unsqueeze(1)
        return (x - torch.median(x, dim=1, keepdim=True)[0]) / denom


def _validate_input(
    configs: Dict[str, Any],
    networks: List[TorchModule],
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """
    Validate the input batch for the predict function.

    Parameters
    ----------
    configs : dict
        Configuration dictionary.
    networks : list
        List of TorchModule instances.
    batch : tuple
        Tuple containing RF and BPM data tensors.

    Raises
    ------
    ValueError
        If input validation fails.
    TypeError
        If input types are incorrect.
    """
    if len(batch) != 2:
        raise ValueError(
            f"Batch must contain exactly two elements: RF data and BPM data. A length of {len(batch)} was provided."
        )
    if len(networks) != 2:
        raise ValueError(
            f"Networks must contain exactly two TorchModule instances: one for RF and one for BPM. A length of {len(networks)} was provided."
        )
    if not isinstance(networks[0], TorchModule) or not isinstance(
        networks[1], TorchModule
    ):
        raise TypeError(
            f"Networks must be instances of TorchModule. Types provided: {type(networks[0])}, {type(networks[1])}."
        )
    if not isinstance(batch[0], torch.Tensor) or not isinstance(batch[1], torch.Tensor):
        raise TypeError(
            f"Batch elements must be torch.Tensor instances. Types provided: {type(batch[0])}, {type(batch[1])}."
        )
    if batch[0].dim() != 2 or batch[1].dim() != 2:
        raise ValueError(
            f"Batch elements must be 2D tensors (shape: (D, N)). Shape provided: {batch[0].shape}, {batch[1].shape}."
        )
    if batch[0].shape[0] != 1 or batch[1].shape[0] != 8:
        raise ValueError(
            f"RF data must have shape (1, N) and BPM data must have shape (8, N). Shape provided: {batch[0].shape}, {batch[1].shape}."
        )
    if batch[0].shape[1] != batch[1].shape[1]:
        raise ValueError(
            f"RF data and BPM data must have the same number of samples (N). N provided: {batch[0].shape[1]}, {batch[1].shape[1]}."
        )
    if not isinstance(configs, dict):
        raise TypeError(
            f"Configs must be a dictionary. Type provided: {type(configs)}."
        )
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
    if batch[0].numel() == 0 or batch[1].numel() == 0:
        raise ValueError("Input tensors must not be empty.")
    if torch.isnan(batch[0]).any() or torch.isnan(batch[1]).any():
        raise ValueError("Input tensors must not contain NaN values.")
    if batch[0].shape[1] < 1065 or batch[1].shape[1] < 1065:
        # Will raise "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
        raise ValueError(
            f"Input tensors must have at least 1065 samples (N) in the second dimension. Shape provided: {batch[0].shape}, {batch[1].shape}."
        )
