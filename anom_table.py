import threading
import logging
from abc import ABC, abstractmethod
from multiprocessing import Manager

from k2eg.serialization import NTTable

from typing import List, Dict, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TimedDict(ABC):
    default_value = None
    """
        A dictionary that holds values for given keys, with a timer that resets the value to False
        after a set period of time. Anytime the dictionary is modified, it writes the current state
        to K2EG, allowing for real-time monitoring of states for klystron stations.

        This class is thread-safe and can be used in a multi-threaded environment.

        Attributes
        ----------
        data : Dict[str, Any]
            A dictionary that holds values for each key.
        write_to_pv : bool
            A flag indicating whether to write to K2EG.
        timers : Dict[str, threading.Timer]
            A dictionary that holds timers for each key, which reset the value to False after a set period of time..
        lock : threading.Lock
            A lock to ensure thread safety when accessing or modifying the data and timers.
        reset_time : int
            The time in seconds after which the value is reset. Default is 300 seconds (5 minutes).
        logger : logging.Logger
            A logger instance for logging debug messages.
        Methods
        -------
        set_key(key: str, value: Any)
            Sets the value for the given key. If the value is True, starts a timer to reset it to False after 5 minutes.
            If the value is False, cancels any existing timer for that key.
        _reset_key(key: str)
            Resets the value for the given key to False and cancels the timer if it exists.
        get_dict()
            Returns a copy of the current state of the dictionary.
        """

    def __init__(
            self,
            keys: List[str],
            write_to_pv: bool = True,
            queue_inst: Optional["Manager.Queue"] = None,  # instrumentation queue
            reset_time: int = 300,  # Reset time in seconds (5 minutes is default)
            logger: logging.Logger = logger):
        """
        Initializes the TimedBoolDict with the given keys, whether to write to K2EG, and the reset time.

        Parameters
        ----------
        keys: List[str]
            A list of keys for which the boolean values will be stored.
        write_to_pv: bool
            A flag indicating whether to write the anomaly state to K2EG. Default is True.
        queue_inst: Optional["Manager.Queue"]
            A queue which is used to communicate with K2EG. Default is None.
        reset_time : int
            The time in seconds after which the value is reset. Default is 300 seconds (5 minutes).
        logger: logging.Logger
            A logger instance for logging debug messages. Default is the module's logger.
        """
        self.data: Dict[str, bool] = {k: self.default_value for k in keys}
        self.write_to_pv: bool = write_to_pv
        self.queue_inst = queue_inst
        self.reset_time: int = reset_time
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.RLock()
        self.logger = logger
        if self.write_to_pv:
            # Always reset the anomaly state to False at initialization
            write_prediction_to_k2eg(self.data, self.queue_inst)
            self.logger.debug("Reset anomaly state PV to all False at initialization.")

    def set_key(self, key: str, value: bool):
        with self.lock:
            self._set_key(key, value)

    @abstractmethod
    def _set_key(self, key: str, value: Any):
        pass

    @abstractmethod
    def _reset_key(self, key: str):
        pass

    def get_dict(self):
        """
        Returns a copy of the current state of the dictionary.
        This method is thread-safe and returns a snapshot of the current anomaly states for all klystron stations.

        Returns
        -------
        Dict[str, Any]
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
                self.queue_inst.put(None)


class TimedBoolDict(TimedDict):
    default_value = False
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
        The time in seconds after which the value is reset to False if it was set to True. Default is 300 seconds
        (5 minutes).
    logger : logging.Logger
        A logger instance for logging debug messages.
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

    def _set_key(self, key: str, value: bool):
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
        self.logger.debug(f"Setting {key} to 1... Current state dict: \n{dict(self.get_dict())}")
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
            write_prediction_to_k2eg(self.data, self.queue_inst)
        self.logger.debug(f"Set {key} to 1. Current state dict: \n{dict(self.get_dict())}")

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
            self.logger.debug(f"Resetting key {key} to 0... Current state dict: \n{dict(self.get_dict())}")
            self.data[key] = False
            if key in self.timers:
                del self.timers[key]
            if self.write_to_pv:
                write_prediction_to_k2eg(self.data, self.queue_inst)
            self.logger.debug(f"Reset key {key} to 0. Current state dict: \n{dict(self.get_dict())}")


class TimedCountDict(TimedDict):
    default_value = 0

    def set_key(self, key: str, value: int):
        pass

    def _reset_key(self, key: str):
        pass


def write_prediction_to_k2eg(
        anomaly_table: dict[str, Any],
        inst_queue: "Manager.Queue"
) -> None:
    """
    Write the anomaly table to K2EG.

    Parameters
    ----------
    anomaly_table : dict[str, Any]
        The anomaly table to write to K2EG in dictionary form with PV-NAME: Any as the entries.
    inst_queue : "Manager.Queue"
        Instrumentation Queue for communicating with K2EG to set EPICS PVs.
    # """
    inst_queue.put({
        'method': 'update_anomaly_state',
        'data': anomaly_table
    })


def set_anomaly_state(
        anomaly_state: TimedDict,
        klys: str,
        state: Any
) -> dict[str, Any]:
    """
    Set the anomaly state for a given klystron station. This updates the internal state of the anomaly dictionary.

    Parameters
    ----------
    anomaly_dict : TimedDict
        The dictionary holding the anomaly states.
    klys : str
        The name of the klystron station to set the state for.
    state : bool
        The state to set (True for anomalous, False for normal).
    """
    anomaly_state.set_key(klys, state)
    return anomaly_state.get_dict()
