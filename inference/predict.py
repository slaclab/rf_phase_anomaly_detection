"""This is a simplified version of the prediction function from the original code.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_train.py
"""

import os
from operator import itemgetter
from typing import Any, Tuple, Dict, List

import torch
import yaml

from p4p.nt import NTTable
from p4p.client.thread import Context
import k2eg
from k2eg.dml import OperationTimeout
from lume_model.models.torch_module import TorchModule


ROOTDIR = os.path.dirname(os.path.abspath(__file__))


class Predict:
    """Predict class for making predictions using LUME-models."""

    def __init__(self, configs=None, networks=None, write_to_pv=False):
        """Initialize the Predict class.
        Args:
            configs (dict, optional): Configuration dictionary for the prediction.
            networks (list, optional): List of TorchModule instances representing the models.
            write_to_pv (bool, optional): Whether to write the prediction result to a PV. Defaults to False.
        """
        self.configs = load_configs() or configs
        self.networks = load_models() or networks
        self.write_to_pv = write_to_pv
        self.klystrons_dict = load_klystron_configs()

    def predict(
        self,
        rf_input_tensor: torch.Tensor,
        bpm_input_tensor: torch.Tensor,
        rf_station: str,
    ) -> bool:
        """Make predictions using the loaded models and provided a single batch of data.
        Args:
            rf_input_tensor (tensor): Tensor of input data for the first model, with a shape of (D, N), where N is
            the number of samples (1066) and D is the number of RF stations (1).
            bpm_input_tensor (tensor): Tensor of input data for the second model, with a shape of (D, N), where N is
            the number of samples and D (1066) is the number of BPMs (8).
            rf_station (str): The PV name of the RF station to write the prediction result to K2EG.
        Returns:
            bool: Prediction result, True if an anomaly is detected, False otherwise.
        """
        anomalous = predict_label(
            self.configs, self.networks, (rf_input_tensor, bpm_input_tensor)
        )
        if anomalous and self.write_to_pv:
            # Create anomaly table with the given station marked as anomalous
            anomaly_table = create_anomaly_table(rf_station, self.klystrons_dict)
            # Write the prediction result to K2EG
            write_prediction_to_k2eg(anomaly_table)
        return anomalous


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


def load_klystron_configs():
    """Load klystron configurations from a YAML file.
    The configs should have the following keys:
        - klystrons: list of klystron station names.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    with open(ROOTDIR + "/klystrons.yml", "r") as file:
        return yaml.safe_load(file)


def predict_label(
    configs: Dict[str, Any],
    networks: List[TorchModule],
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> bool:
    """Make predictions using the loaded models and provided a single batch of data.

    Args:
        configs (dict): Configuration dictionary for the prediction.
        networks (list): List of TorchModule instances representing the models (length should be 2).
        batch (Tuple): Tuple of input data for the models, should be torch tensors and have a length of 2. The first
        element should be the RF data with a shape of (D, N), where N is the number of samples (1066) and D is the
        number of RFs (1), and the second element should be the BPM data with a shape of (D, N), where D is the number
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


def create_anomaly_table(station: str, klystrons: Dict[str, str]) -> NTTable:
    """Create an anomaly table where all klystron stations are set to False,
     and the given anomalous station is set to True.
    Args:
        station (str): The name of the klystron station to mark as anomalous.
        klystrons (dict): Dictionary containing klystron station names.
            Expected format: {'klystrons': ['station_1', 'station_2', ...]}.
    Returns:
        NTTable: A table with anomaly states for each klystron station.
    """
    klys_list = klystrons["klystrons"]
    # Create a table with anomaly states for each klystron, and mark the given station as anomalous
    # Table format:
    # [
    #     {'station': 'station_1', 'anomaly_state': Bool},
    #     {'station': 'station_2', 'anomaly_state': Bool},
    #     ...
    # ]
    # where each dict is a row and its keys are columns.
    anomaly_table = [
        {"station": klys, "anomaly_state": True if klys == station else False}
        for klys in klys_list
    ]
    # Generate output format.
    table_format = NTTable([("station", "s"), ("anomaly_state", "?")])
    return table_format.wrap(anomaly_table)


def write_prediction_to_p4p_sim(anomaly_table: NTTable):
    """For testing purposes, write the anomaly table to a simulated server.
    Args:
        anomaly_table (NTTable): The anomaly table to write to K2EG.
    """
    context = Context()
    anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
    context.put(anomaly_pv, anomaly_table)


def write_prediction_to_k2eg(anomaly_table: NTTable):
    """Write the anomaly table to K2EG.
    Args:
        anomaly_table (NTTable): The anomaly table to write to K2EG.
    """
    anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
    k2eg_client = k2eg.dml("rf-phase-ad", "app-three")
    try:
        k2eg_client.put(f"pva://{anomaly_pv}", anomaly_table, 5.0)
        k2eg_client.close()
    except Exception as e:
        k2eg_client.close()
        if isinstance(e, OperationTimeout):
            print(f"Operation timed out while writing to {anomaly_pv}.")
        else:
            raise e


def standardize_tensor(x):
    return (x - torch.mean(x)) / torch.std(x)


def _validate_input(configs, networks, batch):
    """Validate the input batch for the predict function."""
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
