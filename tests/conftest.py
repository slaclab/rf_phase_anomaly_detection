import os
import random
from typing import Dict, Any
import yaml
import pytest
import torch


@pytest.fixture(scope="session")
def rootdir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# Predict fixtures
@pytest.fixture(scope="module")
def test_data_set(rootdir):
    try:
        with open(f"{rootdir}/fixtures/test_data.pt", "rb") as f:
            rf_data, bpm_data, pv_name = torch.load(f, weights_only=False)
            test_data_set = (rf_data.numpy(), bpm_data.numpy(), pv_name)
        with open(f"{rootdir}/fixtures/test_data_false.pt", "rb") as f:
            rf_data, bpm_data, pv_name = torch.load(f, weights_only=False)
            test_data_set_false = (rf_data.numpy(), bpm_data.numpy(), pv_name)
        return test_data_set, test_data_set_false
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def configs(rootdir) -> Dict[str, Any]:
    try:
        with open(f"{rootdir}/../inference/configs.yml", "rb") as f:
            configs = yaml.safe_load(f)
        return configs
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def pv_name() -> str:
    return "test_pv_name"


@pytest.fixture(scope="module")
def labels_value() -> list:
    labels = torch.zeros(82).tolist()
    idx = random.randint(0, 81)
    labels[idx] = 1
    return labels
