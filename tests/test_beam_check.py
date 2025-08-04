import pytest
import numpy as np

from beam_check_config import BEAM_RATE_PV
from beam_check import BEAM_CHECK_PVS, do_beam_checks


@pytest.fixture
def make_data_missing_pv():
    data = {}
    for name in BEAM_CHECK_PVS:
        if name != BEAM_RATE_PV:
            data[name] = np.ones(120)
    return data


@pytest.fixture
def make_good_data():
    data = {}
    for name in BEAM_CHECK_PVS:
        data[name] = np.ones(120)
    return data


@pytest.fixture
def make_missing_data():
    data = {}
    for name in BEAM_CHECK_PVS:
        n = np.random.randint(110, 121)
        data[name] = np.ones(n)
    return data


def test_do_beam_checks_with_missing_pv(make_data_missing_pv):
    with pytest.raises(KeyError) as e_info:
        do_beam_checks(make_data_missing_pv)


def test_do_beam_checks_with_good_data(make_good_data):
    do_beam_checks(make_good_data)


def test_do_beam_checks_with_missing_data(make_missing_data):
    with pytest.raises(ValueError) as e_info:
        do_beam_checks(make_missing_data)
