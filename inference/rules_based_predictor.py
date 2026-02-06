import logging
import numpy as np

from inference.base_predictor import BasePredictor

from typing import Any, Optional


# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class RulesBasedPredictor(BasePredictor):
    def __init__(self,
                 logger: Optional[logging.Logger] = logger,
                 ):
        super().__init__(logger=logger)

        self.bpm_threshold = 10.

        self.phase_threshold = 0.999
        self.c = 533
        self.n = 20

    def _predict(self, candidate: dict[str, Any]) -> bool:
        """
        Make predictions using rules developed by Finn on the 2024 data.

        Parameters
        ----------
        candidate : dict[str, Any]
            Dictionary of values created by process_candidate in process_b.  At
            minimum, it must have the following entries:
            rf_input : npt.NDArray[number]
                Numpy array of input data for the first model, with a shape of (D, N),
                where D is the number of RF stations (1) and N is the number of samples
                (1066).
            bpm_input : npt.NDArray[number]
                Numpy array of input data for the second model, with a shape of (D, N),
                where D (1066) is the number of BPMs (8) and N is the number of samples
                (1066).
            rf_pv_name : str
                The PV name of the RF station to write the prediction result to K2EG.
            candidate_timestamp: float
                Timestamp of the prediction.

        Returns
        -------
        bool
            Prediction result, True if an anomaly is detected, False otherwise.
        """
        rf_input = candidate["rf_input"].flatten()  # shape is (1066,)
        bpm_input = candidate["bpm_input"]          # shape is (8, 1066)

        # compute the phase (rf) signal quantities
        mask = np.zeros_like(rf_input, dtype=bool)
        mask[self.c - self.n:self.c + self.n] = True
        u = max(abs(rf_input[mask]))
        v = abs(rf_input[~mask])
        phase_signal = sum(u > v) / sum(~mask)  # float between 0 and 1

        # compute the bpm signal quantities
        bpm_signal = float(np.abs(bpm_input).mean(axis=0).max())

        anomalous = bool(phase_signal > self.phase_threshold and bpm_signal > self.bpm_threshold)
        return anomalous
