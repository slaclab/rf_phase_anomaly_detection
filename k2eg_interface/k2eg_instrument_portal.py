from multiprocessing import Manager
from logging.handlers import QueueHandler

import k2eg
from k2eg.dml import OperationTimeout, OperationError
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
            msg = f"Requested method {method} not implemented; no put performed"
            self.logger.warning(msg)
        else:
            x(data)

    def _put(self, pv_name: str, data: Any):
        try:
            self.dml.put(f"pva://{pv_name}", data, 10.0)
        except Exception as e:
            # TODO: OperationTimeout might need to be switched to a normal TimeoutError
            if isinstance(e, (OperationTimeout, TimeoutError)):
                self.logger.warning(f"Operation timed out while writing to {pv_name}.")
            elif isinstance(e, OperationError):
                self.logger.warning(f"Operation errored while writing to {pv_name}.")
            else:
                raise e

    def update_anomaly_state(
            self,
            anomaly_dict: dict[str, bool]
    ):
        anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
        anomaly_table = create_anomaly_table(anomaly_dict)
        self._put(anomaly_pv, anomaly_table)
        self.update_anomaly_any(anomaly_dict)

    def update_anomaly_any(self, anomaly_dict: dict[str, bool]):
        anom_any_pv = "KLYS:SYS0:1:ANOM_ANY"
        anomaly_any = Scalar(key='value', payload=any(anomaly_dict.values()))
        self._put(anom_any_pv, anomaly_any)

    def update_running_pv(self, is_it_running: bool):
        running_pv = "KLYS:SYS0:1:ANOM_SRV"
        scalar = Scalar(key='value', payload=is_it_running)
        self._put(running_pv, scalar)

    def update_buffer_length(self, buffer_len: int):
        buffer_length_pv = "ANOM:SYS0:1:KAD_BUFFER_LENGTH"
        scalar = Scalar(key='value', payload=buffer_len)
        self._put(buffer_length_pv, scalar)

    def update_cand_count(
            self,
            cand_dict: dict[str, int]
    ):
        cand_count_pv = "ANOM:SYS0:1:KAD_CAND_COUNT"
        cand_table = create_cand_count_table(cand_dict)
        self._put(cand_count_pv, cand_table)

    def update_anom_count(
            self,
            anom_dict: dict[str, int]
    ):
        anom_count_pv = "ANOM:SYS0:1:KAD_ANOM_COUNT"
        anom_table = create_anom_count_table(anom_dict)
        self._put(anom_count_pv, anom_table)


def create_anomaly_table(anom_dict: dict[str, bool]) -> NTTable:
    """
    Create an anomaly table in the expected format from a dictionary of anomaly states.

    Parameters
    ----------
    anom_dict : Dict[str, bool]
        A dictionary where keys are klystron station names and values are their anomaly states (True for anomalous,
        False for normal).

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


def create_cand_count_table(cand_dict: dict[str, int]) -> NTTable:
    """
    Create a candidate count table in the expected format from a dictionary of candidate counts.

    Parameters
    ----------
    cand_dict : Dict[str, int]
        A dictionary where keys are klystron station names and values are the count of anomaly candidates seen.

    Returns
    -------
    NTTable
        A table with counts for each klystron station.
    """
    nt_labels = ["station", "candidate_count"]
    table = NTTable(labels=nt_labels)
    table.set_column("station", list(cand_dict.keys()))
    table.set_column("candidate_count", list(cand_dict.values()))
    return table


def create_anom_count_table(anom_dict: dict[str, int]) -> NTTable:
    """
    Create an anomaly count table in the expected format from a dictionary of anomaly counts.

    Parameters
    ----------
    anom_dict : Dict[str, int]
        A dictionary where keys are klystron station names and values are the count of anomalies seen.

    Returns
    -------
    NTTable
        A table with counts for each klystron station.
    """
    nt_labels = ["station", "anomaly_count"]
    table = NTTable(labels=nt_labels)
    table.set_column("station", list(anom_dict.keys()))
    table.set_column("anomaly_count", list(anom_dict.values()))
    return table