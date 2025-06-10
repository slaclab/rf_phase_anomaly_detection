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
            test_data_set = torch.load(f, weights_only=False)
        return test_data_set
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def expected_output(rootdir):
    try:
        with open(f"{rootdir}/fixtures/output.pt", "rb") as f:
            output = torch.load(f, weights_only=False)
        return output
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
    labels = (torch.zeros(82).tolist())
    idx = random.randint(0, 81)
    labels[idx] = 1
    return labels
