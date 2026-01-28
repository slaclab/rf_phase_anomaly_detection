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
            serialization: str,
    ):
        self.logger.info(f"Received {method} request")
        try:
            x = getattr(self, method)
        except AttributeError:
            self.logger.exception(f"Method {method} not implemented")
            raise
        else:
            # serialize here
            x(data)

    def update_anomaly_state(self, anomaly_table: NTTable):
        anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"

        try:
            self.dml.put(f"pva://{anomaly_pv}", anomaly_table, 10.0)
        except Exception as e:
            # TODO: OperationTimeout might need to be switched to a normal TimeoutError
            if isinstance(e, OperationTimeout):
                print(f"Operation timed out while writing to {anomaly_pv}.")
            else:
                raise e
