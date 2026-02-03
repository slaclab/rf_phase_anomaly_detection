import os
import threading
import logging
import yaml
from time import sleep
from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque
from multiprocessing import Manager

from k2eg.serialization import NTTable

from typing import List, Dict, Optional, Any, Callable


ROOTDIR = os.path.dirname(os.path.abspath(__file__))

# Set up logging
logger = logging.getLogger(__name__)
# handler = logging.StreamHandler()
handler = logging.NullHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TimedDict(ABC):
    default_value = None
    write_to_pv_method = 'default'
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
            write_to_pv: bool = True,
            queue_inst: Optional["Manager.Queue"] = None,  # instrumentation queue
            reset_time: Optional[int] = 300,  # Reset time in seconds (5 minutes is default)
            logger: logging.Logger = logger):
        """
        Initializes the TimedDict, flag whether to write to K2EG, and sets the reset time.
        Uses an queue to communicate with K2EG.  Without this queue, there will be no communication.

        Parameters
        ----------
        write_to_pv: bool
            A flag indicating whether to write the anomaly state to K2EG. Default is True.
        queue_inst: Optional["Manager.Queue"]
            A queue which is used to communicate with K2EG. Default is None.
        reset_time : Optional[int]
            The time in seconds after which the value is reset. Default is 300 seconds (5 minutes).
        logger: Optional[logging.Logger]
            A logger instance for logging debug messages. Default is the module's logger.
        """
        self.data: Dict[str, Any] = {
            format_pv_name_for_table(k): self.default_value for k in load_klystron_configs()
        }
        self.write_to_pv: bool = write_to_pv
        self.queue_inst = queue_inst
        self.reset_time: int = reset_time
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.RLock()
        self.logger = logger
        if queue_inst is None:
            self.logger.warning(f"anom_table {str(self)} did not receive a queue instance, disabling PV writing")
            self.write_to_pv = False
        if self.write_to_pv:
            # Always reset the anomaly state to default at initialization
            write_prediction_to_k2eg(self.get_dict(), self.queue_inst, self.write_to_pv_method)
            self.logger.debug("Reset anomaly state PV to all False at initialization.")

    def set_key(self, key: str, value: Any):
        """
        Set the value for a given key in the dictionary. If `write_to_pv` is True, it writes the current anomaly state to K2EG.

        Parameters
        ----------
        key: str
            The key for which to set the value.
        value: Any
            The value to set for the key.

        Returns
        -------
        None
        """
        with self.lock:
            self._set_key(key, value)

    @abstractmethod
    def _set_key(self, key: str, value: Any):
        pass

    @abstractmethod
    def _reset_key(self, key: str):
        """
        Reset the value for a given key to default. If `write_to_pv` is True, it writes the updated value to K2EG.

        Parameters
        ----------
        key: str
            The key to reset in the dictionary.

        Returns
        -------
        None
        """
        pass

    def get_dict(self):
        """
        Returns a copy of the current state of the dictionary.
        This method is thread-safe and returns a snapshot of the current dictionary for all klystron stations.

        Returns
        -------
        Dict[str, Any]
            A copy of the current state of the dictionary, where keys are klystron station names and values are
            whatever the TimedDict is storing.
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
                pass


class TimedBoolDict(TimedDict):
    default_value = False
    write_to_pv_method = 'update_anomaly_state'
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
        key = format_pv_name_for_table(key)
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
            write_prediction_to_k2eg(self.get_dict(), self.queue_inst, self.write_to_pv_method)
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
        key = format_pv_name_for_table(key)
        with self.lock:
            self.logger.debug(f"Resetting key {key} to 0... Current state dict: \n{dict(self.get_dict())}")
            self.data[key] = self.default_value
            if key in self.timers:
                del self.timers[key]
            if self.write_to_pv:
                write_prediction_to_k2eg(self.get_dict(), self.queue_inst, self.write_to_pv_method)
            self.logger.debug(f"Reset key {key} to 0. Current state dict: \n{dict(self.get_dict())}")


class RepeatingTask:
    def __init__(
            self,
            executable: Optional[Callable] = None,
            sleep_time: Optional[float] = 2,
            sleep_time_step: Optional[float] = 1,
    ):
        self.executable = executable
        assert sleep_time_step <= sleep_time, 'sleep_time_step must be <= sleep_time'
        self.sleep_time = sleep_time
        self.sleep_time_step = sleep_time_step
        self.keep_running = False
    def __call__(self):
        self.keep_running = True
        sleep_timer = 0
        while self.keep_running:
            sleep(self.sleep_time_step)
            sleep_timer += self.sleep_time_step
            if sleep_timer >= self.sleep_time:
                self.execute()
                sleep_timer = 0
    def execute(self):
        if callable(self.executable):
            self.executable()


class StoppableThread(threading.Thread):
    """
    Thread class with a stop() method.
    The thread itself has to check regularly for the stopped() condition.
    """
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()
    def stop(self):
        self._stop_event.set()
        try:
            self._target.keep_running = False
        except AttributeError:
            pass
    def cancel(self):
        self.stop()
    @property
    def stopped(self):
        return self._stop_event.is_set()


class TimedCountDict(TimedDict):
    def _set_key(self, key: str, value: Optional[float] = None):
        if self.timers == {}:
            sts = min(self.reset_time / 10, 1)
            y = RepeatingTask(self.drop_old_with_lock, self.reset_time, sleep_time_step=sts)
            self.timers['all'] = StoppableThread(target=y)
            self.timers['all'].start()

        key = format_pv_name_for_table(key)
        if value is None:
            value = datetime.now().timestamp()
        self.logger.debug(f"Incrementing {key} by 1... Current state dict: \n{dict(self.get_dict())}")

        dd = self.data[key]
        if dd is None:
            dd = deque(maxlen=3000)
        dd.append(value)
        self.data[key] = dd
        self.drop_old()  # calls write_prediction_to_k2eg
        self.logger.debug(f"Incrementing {key} to 1. Current state dict: \n{dict(self.get_dict())}")

    def _reset_key(self, key: str):
        key = format_pv_name_for_table(key)
        with self.lock:
            self.logger.debug(f"Resetting key {key} ... Current state dict: \n{dict(self.get_dict())}")
            self.data[key] = self.default_value
            self.drop_old()  # calls write_prediction_to_k2eg
            self.logger.debug(f"Reset key {key} to 0. Current state dict: \n{dict(self.get_dict())}")

    def get_dict(self) -> dict:
        dd = super().get_dict()
        return {k: (len(v) if v is not None else 0) for k, v in dd.items()}

    def drop_old(self) -> None:
        """
        Looks for klystrons with values that are more than self.reset_time into the past and drops them.
        """
        now = datetime.now().timestamp()
        for v in self.data.values():
            if v is not None:
                while len(v) > 0 and v[0] <= now - self.reset_time:
                    v.popleft()
        if self.write_to_pv:
            write_prediction_to_k2eg(self.get_dict(), self.queue_inst, self.write_to_pv_method)

    def drop_old_with_lock(self) -> None:
        with self.lock:
            self.drop_old()


class TimedCandCountDict(TimedCountDict):
    write_to_pv_method = 'update_cand_count'

class TimedAnomCountDict(TimedCountDict):
    write_to_pv_method = 'update_anom_count'


def write_prediction_to_k2eg(
        anomaly_table: dict[str, Any],
        inst_queue: "Manager.Queue",
        method: str = "default"
) -> None:
    """
    Write the anomaly table to K2EG.

    Parameters
    ----------
    anomaly_table : dict[str, Any]
        The anomaly table to write to K2EG in dictionary form with PV-NAME: Any as the entries.
    inst_queue : "Manager.Queue"
        Instrumentation Queue for communicating with K2EG to set EPICS PVs (K2EGInstrumentPortal).
    method : str
        The method in the communication handler (K2EGInstrumentPortal) to call.  Default will typically not work.
    # """
    inst_queue.put({
        'method': method,
        'data': anomaly_table
    })


def set_timed_dict(
        anomaly_state: TimedDict,
        klys: str,
        state: Any
) -> dict[str, Any]:
    """
    Set a timed dict for a given klystron station. This updates the internal state of the dictionary.

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


def format_pv_name_for_table(name: str) -> str:
    s = name.split(':')
    return '_'.join(s[:3]).lower()


def load_klystron_configs() -> List:
    """
    Load klystron configurations from a YAML file.

    The configs should have the following keys:
        - klystrons: list of klystron station names.

    Returns
    -------
    List
        List of klystron names loaded from the YAML file.
    """
    with open(ROOTDIR + "/resources/klystrons.yml", "r") as file:
        return yaml.safe_load(file)["klystrons"]