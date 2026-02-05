import logging
from datetime import datetime
from abc import ABC, abstractmethod

from run_config import NANOSECS_IN_1_SEC

from typing import Any, Optional


# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BasePredictor(ABC):
    def __init__(self,
                 logger: Optional[logging.Logger] = logger,
                 ):
        self.logger = logger
        self.logger.info(f"({str(self)}) Starting predictor")

    def __str__(self) -> str:
        return self.__class__.__name__

    def predict(self, candidate: dict[str, Any]) -> bool:
        """
        Make predictions using the loaded models and provided a single batch of data.

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
        anomalous = self._predict(candidate=candidate)
        rf_pv_name = candidate["rf_pv_name"]
        candidate_timestamp = candidate["candidate_timestamp"]
        candidate_time = str(datetime.fromtimestamp(candidate_timestamp / NANOSECS_IN_1_SEC))
        if anomalous:
            log_method = self.logger.info
            log_start = "Anomaly"

        else:
            log_method = self.logger.debug
            log_start = "No anomaly"
        msg = f"({str(self)}) {log_start} detected for {rf_pv_name} at time {candidate_time} "
        msg += f"({candidate_timestamp} ns)."
        log_method(msg)

        return anomalous

    @abstractmethod
    def _predict(self, candidate: dict[str, Any]) -> bool:
        pass

    def shut_down(self):
        """
        Clean up any configuration used by the predictor.
        """
        self.logger.info(f"({str(self)}) Shutting down predictor")
