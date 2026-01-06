"""Test for anomaly state management in Predict class.
If you want to run online (writing to PV), set `write_to_pv=True` in the Predict constructor.
Run from the project root directory with:
`python -m tests.anom_table_k2eg_test`
"""

import sys

if "pytest" in sys.modules:
    import pytest

    pytest.skip("Not a test module", allow_module_level=True)

import os
import logging
import time
import torch

from inference.coad_predictor import COADPredictor

logging.basicConfig(level=logging.DEBUG)


def get_data():
    """
    Load test data from fixtures.
    """
    rootdir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{rootdir}/fixtures/test_data.pt", "rb") as f:
        rf_data, bpm_data, pv_name = torch.load(f, weights_only=False)
        test_data_set_true = (rf_data.numpy(), bpm_data.numpy(), pv_name)
    with open(f"{rootdir}/fixtures/test_data_false.pt", "rb") as f:
        rf_data, bpm_data, pv_name = torch.load(f, weights_only=False)
        test_data_set_false = (rf_data.numpy(), bpm_data.numpy(), pv_name)
    return test_data_set_true, test_data_set_false


def main():
    """
    Test the anomaly state management in Predict class.
    This function simulates setting and resetting anomaly states for multiple stations
    over a period of time, checking the expected behavior of the anomaly state dictionary.
    """
    pred = COADPredictor(write_to_pv=False)
    # Silence lume-model out of range warnings
    pred.networks[0].model.input_validation_config = {n: "none" for n in pred.networks[0].model.input_names}
    pred.networks[1].model.input_validation_config = {n: "none" for n in pred.networks[1].model.input_names}
    pred.anom_state_dict.reset_time = 50  # seconds, setting a shorter reset time for testing

    # Five random keys
    stations = pred.klystrons_list[0:4]
    print("Stations:", stations)

    # Get test data
    test_data_set_true, test_data_set_false = get_data()

    data_true = {
        "anomaly_timestamp": 1000,  # Example timestamp
        "rf_input": test_data_set_true[0],
        "bpm_input": test_data_set_true[1],
        "rf_pv_name": None,  # Will be set later
    }

    data_false = {
        "anomaly_timestamp": 1000,  # Example timestamp
        "rf_input": test_data_set_false[0],
        "bpm_input": test_data_set_false[1],
        "rf_pv_name": None,
    }

    # Set the anomaly state for each station

    # Set anomaly state for the first station
    data_true["rf_pv_name"] = stations[0]
    data_false["rf_pv_name"] = stations[0]
    pred.predict(**data_true)
    time.sleep(5)
    # Get no anomaly results for the first station, dict should not change True state
    pred.predict(**data_false)
    time.sleep(20)

    print(f"After 25s, {stations[0]} should be True\n")

    # Set anomaly state for the second station
    data_true["rf_pv_name"] = stations[1]
    pred.predict(**data_true)
    time.sleep(10)

    print(f"After 35s, {stations[0]} and {stations[1]} should be True\n")

    # Set anomaly state for the third station
    data_true["rf_pv_name"] = stations[2]
    pred.predict(**data_true)
    time.sleep(30)

    print(f"After 65s, {stations[1]} and {stations[2]} should be True\n")

    # Set the third station to True again, should not reset after 50s
    data_true["rf_pv_name"] = stations[2]
    pred.predict(**data_true)
    time.sleep(9)

    print(f"After 74s, {stations[1]} and {stations[2]} should be True\n")

    # Set the second station to True again, should not reset after 50s
    data_true["rf_pv_name"] = stations[1]
    pred.predict(**data_true)
    time.sleep(1)

    print(f"After 75s, {stations[1]} and {stations[2]} should be True\n")

    time.sleep(41)
    print(f"After 116s, {stations[1]} should be True\n")

    time.sleep(8)
    print("All stations should be False")

    pred.anom_state_dict.shut_down()  # clean up the timers


if __name__ == "__main__":
    main()
