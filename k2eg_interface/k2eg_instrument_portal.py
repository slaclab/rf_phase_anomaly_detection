from multiprocessing import Manager
from logging.handlers import QueueHandler

import k2eg
from k2eg.dml import OperationTimeout
from k2eg.dml import dml as k2eg_dml
from k2eg.serialization import Scalar, Vector, NTTable

from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Any


# APP_NAME appears to be important to accessing the kafka server
# the bottom APP_NAME works, but the top one does not
# someone needs to configure kafka
# APP_NAME = 'rf-phase-anomaly-detection'
APP_NAME = "app-phase-anomaly-detection-put"


class K2EGInstrumentPortal:
    def __init__(
            self,
            logging_kwargs: dict = default_logging_kwargs
    ):
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "K2EGPutPortal"
        self.logger = None
        self.dml = None
        self.is_running = False

    def __enter__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        self.logger.info("Starting put portal for K2EG")

        try:
            self.dml = k2eg.dml("k2eg", APP_NAME)
        except TimeoutError:
            self.logger.exception("Failed to start k2eg dml instance")
            raise
        else:
            self.is_running = True
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dml.close()
        self.is_running = False
        self.logger.info("Shutdown put portal")

    def __call__(
            self,
            method: str,
            data: Any,
    ):
        self.logger.info(f"Received {method} request")
        try:
            x = getattr(self, method)
        except AttributeError:
            self.logger.exception(f"Method {method} not implemented")
            raise
        else:
            x(data)

    def update_anomaly_state(
            self,
            anomaly_dict: dict[str, bool]
    ):
        anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
        anomaly_table = create_anomaly_table(anomaly_dict)

        try:
            self.dml.put(f"pva://{anomaly_pv}", anomaly_table, 10.0)
        except Exception as e:
            # TODO: OperationTimeout might need to be switched to a normal TimeoutError
            if isinstance(e, OperationTimeout):
                print(f"Operation timed out while writing to {anomaly_pv}.")
            else:
                raise e


def create_anomaly_table(anom_dict: dict[str, bool]) -> NTTable:
    """
    Create an anomaly table in the expected format from a dictionary of anomaly states.

    Parameters
    ----------
    anom_dict : Dict[str, bool]
        A dictionary where keys are klystron station names and values are their anomaly states (True for anomalous, False for normal).

    Returns
    -------
    NTTable
        A table with anomaly states for each klystron station.
    """
    nt_labels = ["station", "anomaly_state"]
    table = NTTable(labels=nt_labels)
    table.set_column("station", list(anom_dict.keys()))
    table.set_column("anomaly_state", list(anom_dict.values()))
    return table