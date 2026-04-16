from p4p.nt import NTScalar, NTTable
from p4p.client.thread import Context

from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Any, Optional


class P4PInstrumentPortal:
    """Generic interface for interacting with EPICS PVs using p4p."""

    def __init__(
            self,
            protocol: str = 'pva',
            logging_kwargs: dict = default_logging_kwargs
    ):
        """
        Initialize the P4P interface.

        Parameters
        ----------
        protocol : str, optional
            Protocol to use ('pva' for PVAccess or 'ca' for Channel Access), by default 'pva'
        """
        self.protocol = protocol

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "P4PPutPortal"
        self.logger = None

        self.is_running = False
        self.ctx = None

    def _put(self, pv_name: str, value: Any, timeout: Optional[float] = 5.0) -> None:
        """
        Write a value to a PV.

        Parameters
        ----------
        pv_name : str
            Name of the process variable
        value : Any
            Value to write (can be any type supported by the PV)
        timeout : float, optional
            Operation timeout in seconds, by default 5.0
        """
        try:
            self.ctx.put(pv_name, value, timeout=timeout)
        except TimeoutError:
            msg = f"P4PInstrumentPortal put to {pv_name} timed out; "
            msg += "it is likely the mailbox is not reachable, shutting down"
            self.logger.exception(msg)
            raise

    def _get(self, pv_name: str, timeout: Optional[float] = 5.0) -> Any:
        """
        Read a value from a PV.

        Parameters
        ----------
        pv_name : str
            Name of the process variable
        timeout : float, optional
            Operation timeout in seconds, by default 5.0

        Returns
        -------
        Any
            The current value of the PV
        """
        return self.ctx.get(pv_name, timeout=timeout)

    def __call__(
            self,
            method: str,
            data: Any,
    ):
        self.logger.debug(f"Received {method} request")
        try:
            x = getattr(self, method)
        except AttributeError:
            msg = f"Requested method {method} not implemented; no put performed"
            self.logger.warning(msg)
        except TimeoutError:
            msg = f"Requested method {method} timed out; no put performed"
            self.logger.warning(msg)
        else:
            x(data)

    def close(self) -> None:
        """Close the context and clean up resources."""
        self.ctx.close()

    def __enter__(self):
        """Context manager entry."""
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        self.logger.info("Starting put portal for P4P")

        try:
            self.ctx = Context(self.protocol)
        except TimeoutError:
            self.logger.exception("Failed to start p4p Context instance")
            raise
        else:
            self.is_running = True
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        self.is_running = False
        self.logger.info("Shutdown put portal")

    def update_anomaly_state(
            self,
            anomaly_dict: dict[str, bool]
    ):
        anomaly_table = create_anomaly_table(anomaly_dict)
        # the new PV name:
        anomaly_pv = "ANOM:SYS0:1:KAD_STATES"
        self._put(anomaly_pv, anomaly_table)
        self.update_anomaly_any(anomaly_dict)
        # the old PV name (to be deleted):
        anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
        self._put(anomaly_pv, anomaly_table)

    def update_anomaly_any(self, anomaly_dict: dict[str, bool]):
        scalar_type = NTScalar("?")  # bool scalar type
        anomaly_any = scalar_type.wrap(any(anomaly_dict.values()))
        # the new PV name:
        anom_any_pv = "ANOM:SYS0:1:KAD_ANOM_ANY"
        self._put(anom_any_pv, anomaly_any)
        # the old PV name (to be deleted):
        anom_any_pv = "KLYS:SYS0:1:ANOM_ANY"
        self._put(anom_any_pv, anomaly_any)

    def update_running_pv(self, is_it_running: bool):
        scalar_type = NTScalar("?")  # bool scalar type
        scalar = scalar_type.wrap(is_it_running)
        # the new PV name
        running_pv = "ANOM:SYS0:1:KAD_RUNNING"
        self._put(running_pv, scalar)
        # the old PV name (to be deleted):
        running_pv = "KLYS:SYS0:1:ANOM_SRV"
        self._put(running_pv, scalar)

    def update_buffer_length(self, buffer_len: int):
        buffer_length_pv = "ANOM:SYS0:1:KAD_BUFFER_LENGTH"
        scalar_type = NTScalar("L")
        scalar = scalar_type.wrap(buffer_len)
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
    table = NTTable([("station", "s"), ("anomaly_state", "?")])
    anomaly_table = [{"station": klys, "anomaly_state": status} for klys, status in anom_dict.items()]
    anomaly_table = table.wrap(anomaly_table)
    return anomaly_table


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
    table = NTTable([("station", "s"), ("candidate_count", "L")])
    cand_table = [{"station": klys, "candidate_count": count} for klys, count in cand_dict.items()]
    cand_table = table.wrap(cand_table)
    return cand_table


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
    table = NTTable([("station", "s"), ("anomaly_count", "L")])
    anom_table = [{"station": klys, "anomaly_count": count} for klys, count in anom_dict.items()]
    anom_table = table.wrap(anom_table)
    return anom_table
