import pytest
import numpy as np

from k2eg_process import read_pv_list_from_file
from beam_check_config import CANDIDATE_LOOKBACK_WINDOW_LENGTH, FEEDBACK_STATIONS
from anomaly_candidate import find_most_anomalous_rf_station


RF_PV_NAMES = [n for n in read_pv_list_from_file('resources/pv_list.txt') if n.endswith('PHAS_FASTBR')]

@pytest.fixture
def make_random_data():
    rng = np.random.default_rng(12345)
    return rng.normal(size=(CANDIDATE_LOOKBACK_WINDOW_LENGTH, len(RF_PV_NAMES)))


@pytest.fixture
def make_single_anomaly_data():
    rng = np.random.default_rng(12345)
    index = rng.choice(len(RF_PV_NAMES))
    clwl = CANDIDATE_LOOKBACK_WINDOW_LENGTH
    data = np.zeros((clwl, len(RF_PV_NAMES)))
    data[:, index] = (np.arange(clwl) > 10).astype(float) * 5.0
    return index, data


@pytest.fixture
def make_feedback_anomaly_data():
    rng = np.random.default_rng(12345)
    index = 27  # KLYS:LI24:11:PHAS_FASTBR
    clwl = CANDIDATE_LOOKBACK_WINDOW_LENGTH
    data = rng.random((clwl, len(RF_PV_NAMES)))
    data[:, index] = (np.arange(clwl) > 10).astype(float) * 5.0
    return index, data


@pytest.fixture
def make_feedback_only_anomaly_data():
    rng = np.random.default_rng(12345)
    clwl = CANDIDATE_LOOKBACK_WINDOW_LENGTH
    data = rng.random((clwl, len(FEEDBACK_STATIONS)))
    return 0, data


def test_on_random_data(make_random_data):
    rf_name, deviation, system = find_most_anomalous_rf_station(
        make_random_data, rf_pv_names=RF_PV_NAMES)
    assert system  # should be true


def test_on_single_anomaly_data(make_single_anomaly_data):
    index, data = make_single_anomaly_data
    rf_name, deviation, system = find_most_anomalous_rf_station(data, rf_pv_names=RF_PV_NAMES)
    assert rf_name == RF_PV_NAMES[index]
    assert deviation == 5.0
    assert not system  # should be false


def test_on_feedback_anomaly_data(make_feedback_anomaly_data):
    index, data = make_feedback_anomaly_data
    rf_name, deviation, system = find_most_anomalous_rf_station(data, rf_pv_names=RF_PV_NAMES)
    assert rf_name != RF_PV_NAMES[index]  # should not match
    assert deviation != 5.0  # should not match
    assert not system  # should be false


def test_on_feedback_only_anomaly_data(make_feedback_only_anomaly_data):
    index, data = make_feedback_only_anomaly_data
    rf_name, deviation, system = find_most_anomalous_rf_station(
        data,
        rf_pv_names=FEEDBACK_STATIONS
    )
    assert rf_name == ''
    assert deviation == 0.0
    assert not system  # should be false
