from typing import List, Dict
import threading
import logging

from p4p.nt import NTTable
from p4p.client.thread import Context
import k2eg
from k2eg.dml import OperationTimeout
from k2eg.dml import dml as k2eg_dml

logger = logging.getLogger(__name__)


class TimedBoolDict:
    """
    A dictionary that holds boolean values for given keys, with a timer that resets the value to False
    after 5 minutes if the value is set to True. Anytime the dictionary is modified, it writes the current state
    to K2EG, allowing for real-time monitoring of anomaly states for klystron stations.

    This class is thread-safe and can be used in a multi-threaded environment.

    Attributes
    ----------
    data : Dict[str, bool]
        A dictionary that holds boolean values for each key.
    write_to_pv : bool
        A flag indicating whether to write the anomaly state to K2EG.
    timers : Dict[str, threading.Timer]
        A dictionary that holds timers for each key, which reset the value to False after 5 minutes.
    lock : threading.Lock
        A lock to ensure thread safety when accessing or modifying the data and timers.
    reset_time : int
        The time in seconds after which the value is reset to False if it was set to True. Default is 300 seconds (5 minutes).
    k2eg_client : k2eg_dml
        The K2EG client used to write the anomaly state to K2EG if `write_to_pv` is True.
    Methods
    -------
    set_key(key: str, value: bool)
        Sets the value for the given key. If the value is True, starts a timer to reset it to False after 5 minutes.
        If the value is False, cancels any existing timer for that key.
    _reset_key(key: str)
        Resets the value for the given key to False and cancels the timer if it exists.
    get_dict()
        Returns a copy of the current state of the dictionary.
    """
    def __init__(self, keys: List[str], write_to_pv: bool = True, reset_time: int = 300):
        self.data: Dict[str, bool] = {k: False for k in keys}
        self.write_to_pv: bool = write_to_pv
        self.reset_time: int = reset_time  # Reset time in seconds (5 minutes is default)
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.RLock()
        if self.write_to_pv:
            self.k2eg_client = k2eg.dml("rf-phase-ad", "app-three")

    def set_key(self, key: str, value: bool):
        """
        Set the value for a given key in the dictionary. If the value is True, starts a timer to reset it to False
        after 5 minutes. If the value is False, cancels any existing timer for that key. If `write_to_pv` is True,
        it writes the current anomaly state to K2EG.

        Parameters
        ----------
        key: str
            The key for which to set the value.
        value: bool
            The value to set for the key. If True, starts a timer to reset it to False after 5 minutes; if False,
            cancels any existing timer for that key and sets the value to False in the dictionary.

        Returns
        -------
        None
        """
        with self.lock:
            self.data[key] = value
            if value:
                # Cancel existing timer if present
                if key in self.timers:
                    self.timers[key].cancel()
                # Start/reset timer for 5 minutes
                timer = threading.Timer(self.reset_time, self._reset_key, args=(key,))
                self.timers[key] = timer
                timer.start()
            else:
                # Cancel timer if value set to 0
                if key in self.timers:
                    self.timers[key].cancel()
                    del self.timers[key]
            if self.write_to_pv:
                anomaly_table = create_anomaly_table(self.data)
                write_prediction_to_k2eg(anomaly_table, self.k2eg_client)
            logger.debug(f"Current state dict: {dict(self.get_dict())}")

    def _reset_key(self, key: str):
        """
        Reset the value for a given key to False and cancel any existing timer for that key. If `write_to_pv` is True,
        it writes the updated anomaly state to K2EG.

        Parameters
        ----------
        key: str
            The key to reset in the dictionary.

        Returns
        -------
        None
        """
        with self.lock:
            self.data[key] = False
            if key in self.timers:
                del self.timers[key]
            if self.write_to_pv:
                anomaly_table = create_anomaly_table(self.data)
                write_prediction_to_k2eg(anomaly_table, self.k2eg_client)
            logger.debug(f"Current state dict: {dict(self.get_dict())}")

    def get_dict(self):
        """
        Returns a copy of the current state of the dictionary.
        This method is thread-safe and returns a snapshot of the current anomaly states for all klystron stations.

        Returns
        -------
        Dict[str, bool]
            A copy of the current state of the dictionary, where keys are klystron station names and values are their
            anomaly states (True for anomalous, False for normal).
        """
        with self.lock:
            return dict(self.data)

    def shut_down(self):
        """
        Cancel all timers and close the K2EG client connection if `write_to_pv` is True.
        This method should be called when the application is shutting down to clean up resources.

        Returns
        -------
        None
        """
        with self.lock:
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()
            if self.write_to_pv:
                self.k2eg_client.close()


def create_anomaly_table(anom_dict: Dict[str, bool]) -> NTTable:
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
    # Create a table with anomaly states for each klystron, and mark the given station as anomalous
    # Table format:
    # [
    #     {'station': 'station_1', 'anomaly_state': Bool},
    #     {'station': 'station_2', 'anomaly_state': Bool},
    #     ...
    # ]
    # where each dict is a row and its keys are columns.
    anomaly_table = [
        {"station": klys, "anomaly_state": state}
        for klys, state in anom_dict.items()
    ]
    # Generate output format.
    table_format = NTTable([("station", "s"), ("anomaly_state", "?")])
    return table_format.wrap(anomaly_table)


def write_prediction_to_p4p_sim(anomaly_table: NTTable) -> None:
    """
    For testing purposes, write the anomaly table to a simulated server.

    Parameters
    ----------
    anomaly_table : NTTable
        The anomaly table to write to K2EG.
    """
    context = Context()
    anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"
    context.put(anomaly_pv, anomaly_table)


def write_prediction_to_k2eg(anomaly_table: NTTable, k2eg_client: k2eg_dml) -> None:
    """
    Write the anomaly table to K2EG.

    Parameters
    ----------
    anomaly_table : NTTable
        The anomaly table to write to K2EG.
    k2eg_client : k2eg_dml
        The K2EG client to use for writing the anomaly table.
    """
    anomaly_pv = "KLYS:SYS0:1:ANOM_STATES"

    try:
        k2eg_client.put(f"pva://{anomaly_pv}", anomaly_table, 10.0)
    except Exception as e:
        if isinstance(e, OperationTimeout):
            print(f"Operation timed out while writing to {anomaly_pv}.")
        else:
            raise e


def set_anomaly_state(
    anomaly_state: TimedBoolDict, klys: str, state: bool
) -> None:
    """
    Set the anomaly state for a given klystron station. This updates the internal state of the anomaly dictionary.

    Parameters
    ----------
    anomaly_dict : TimedBoolDict
        The dictionary holding the anomaly states.
    klys : str
        The name of the klystron station to set the state for.
    state : bool
        The state to set (True for anomalous, False for normal).
    """
    anomaly_state.set_key(klys, state)
    anomaly_dict = anomaly_state.get_dict()
    return anomaly_dict