from typing import Dict, Any, Tuple, List
from unittest import mock
import traceback
import os
import sys

import numpy.typing as npt
from numpy import number
from numpy import zeros
import pytest

import k2eg
from k2eg.dml import OperationTimeout
from inference.predict import Predict


class TestPredict:
    """
    Test suite for the Predict class and its integration with K2EG.

    Methods
    -------
    test_predict(test_data_set, configs)
        Test predictions for both True and False cases.
    test_predict_with_wrong_type()
        Test prediction with incorrect input types.
    test_predict_with_empty_input()
        Test prediction with empty or NaN input.
    test_predict_write_table_to_k2eg(test_data_set, rootdir)
        Test writing prediction results to K2EG.
    test_writing_to_k2eg(pv_name, labels_value, rootdir)
        Test direct writing to K2EG.
    """

    timestamp = 1000  # unused for now

    def test_predict(
        self,
        test_data_set: List[Tuple[npt.NDArray[number], npt.NDArray[number], str]],
        configs: Dict[str, Any],
    ) -> None:
        """
        Test predictions for both True and False cases.

        Parameters
        ----------
        test_data_set : list of tuple
            List containing tuples of (rf_data, bpm_data, pv_name).
        configs : dict
            Configuration dictionary for the predictor.
        """
        predictor = Predict(write_to_pv=False)

        # Get data that should return a prediction of True
        rf_data, bpm_data, pv_name = test_data_set[0]
        # Run prediction
        predictions = predictor.predict(rf_data, bpm_data, pv_name, self.timestamp)
        # Check configs loaded correctly
        assert predictor.configs == configs
        # Check if predictions are of the expected shape
        assert isinstance(predictions, bool)
        # Check if predictions match the expected output data
        assert predictions

        # Get data that should return a prediction of False
        rf_data, bpm_data, pv_name = test_data_set[1]
        # Run prediction
        predictions = predictor.predict(rf_data, bpm_data, pv_name, self.timestamp)
        # Check configs loaded correctly
        assert predictor.configs == configs
        # Check if predictions are of the expected shape
        assert isinstance(predictions, bool)
        # Check if predictions match the expected output data
        assert not predictions

        predictor.anom_state_dict.shut_down()

    def test_predict_with_wrong_shape(self) -> None:
        """
        Test prediction with incorrect input shapes.

        Raises
        ------
        ValueError
            If input types are the correct shapes.
        """
        predictor = Predict(write_to_pv=False)
        with pytest.raises(ValueError):
            predictor.predict([1], [2], [3], self.timestamp)
        predictor.anom_state_dict.shut_down()

    def test_predict_with_empty_input(self) -> None:
        """
        Test prediction with empty or NaN input tensors.

        Raises
        ------
        ValueError
            If input tensors are empty or contain NaN values.
        """
        predictor = Predict(write_to_pv=False)
        with pytest.raises(ValueError):
            # add nans to the input tensors
            in1 = zeros((1, 1066))
            in2 = zeros((8, 1066))
            in1[0, 0] = float("nan")
            in2[0, 0] = float("nan")
            predictor.predict(in1, in2, "pv", self.timestamp)
        predictor.anom_state_dict.shut_down()

    def test_predict_write_table_to_k2eg(
        self,
        test_data_set: List[Tuple[npt.NDArray[number], npt.NDArray[number], str]],
        rootdir: str,
    ) -> None:
        """
        Test writing prediction results to K2EG.

        Parameters
        ----------
        test_data_set : list of tuple
            List containing tuples of (rf_data, bpm_data, pv_name).
        rootdir : str
            Root directory for configuration.
        """
        # NOTE: May need to delete/adjust this test once tested and deployed
        # Set up k2eg config
        k2eg_conf_dir = f"{rootdir}/../config/"
        with mock.patch.dict(
            os.environ, {"K2EG_PYTHON_CONFIGURATION_PATH_FOLDER": k2eg_conf_dir}
        ):
            predictor = Predict(write_to_pv=True)
            # Get data that should return a prediction of True
            rf_data, bpm_data, _ = test_data_set[0]
            pv_name = "klys_li21_61"
            predictions = predictor.predict(rf_data, bpm_data, pv_name, self.timestamp)
        assert predictions
        predictor.anom_state_dict.shut_down()

    def test_writing_to_k2eg(self, pv_name, labels_value, rootdir):
        """
        Test direct writing to K2EG.

        Parameters
        ----------
        pv_name : str
            PV name to write to.
        labels_value : Any
            Value to write.
        rootdir : str
            Root directory for configuration.

        Skips test if K2EG is not available or times out.
        """
        # NOTE: May need to delete/adjust this test once tested and deployed
        k2eg_conf_dir = f"{rootdir}/../config/"
        with mock.patch.dict(
            os.environ, {"K2EG_PYTHON_CONFIGURATION_PATH_FOLDER": k2eg_conf_dir}
        ):
            try:
                k2eg_client = k2eg.dml("rf-phase-ad", "app-three")
                k2eg_client.put(f"pva://{pv_name}", labels_value, 1.0)
            except ValueError:
                trb = traceback.format_exc()
                if "[kafka_broker_url] Kafka broker url is mandatory" in str(trb):
                    k2eg_client.close()
                    pytest.skip("k2eg client not available or failed to connect")
            except TimeoutError:
                trb = traceback.format_exc()
                if "Function timed out" in str(trb):
                    pytest.skip("k2eg client timed out")
            except OperationTimeout:
                trb = traceback.format_exc()
                if "Timeout during start" in str(trb):
                    pytest.skip("k2eg client operation timed out")

            k2eg_client.close()
