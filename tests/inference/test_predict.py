from typing import Dict, Any
import pytest
import torch

from inference.predict import Predict


class TestPredict:
    def test_predict(
        self,
        test_data_set,
        configs: Dict[str, Any],
        expected_output: torch.Tensor,
    ):
        predictor = Predict()
        rf_data, bpm_data, pv_names = test_data_set

        # Run prediction
        predictions = predictor.predict([rf_data, bpm_data])

        # Check configs loaded correctly
        assert predictor.configs == configs
        # Check if predictions are of the expected shape
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape[0] == len(rf_data)

        # Check if predictions match the expected output data
        assert all(predictions == expected_output)

    def test_predict_with_invalid_input(self):
        pass  # TODO: add validation to handle invalid input

    def test_predict_with_empty_input(self):
        predictor = Predict()
        with pytest.raises(IndexError):  # TODO: add validation to handle empty input
            predictor.predict([])
