from typing import Dict, Any
import traceback
import pytest
import torch

import k2eg
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

    def test_writing_predictions_to_k3eg(self, pv_name, labels_value):
        try:
            k2eg_client = k2eg.dml('rf-phase-ad', 'app-three')
            k2eg_client.put(
                f'pva://{pv_name}',
                labels_value,
                5.0
            )
        except ValueError:
            trb = traceback.format_exc()
            if "[kafka_broker_url] Kafka broker url is mandatory" in str(trb):
                pytest.skip("k2eg client not available or failed to connect")
