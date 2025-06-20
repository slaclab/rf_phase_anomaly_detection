import time
from multiprocessing import Manager
import k2eg
from k2eg.broker import SnapshotProperties, SnapshotType
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional, Callable


# APP_NAME appears to be important to accessing the kafka server
# the bottom APP_NAME works, but the top one does not
# someone needs to configure kafka
# APP_NAME = 'rf-phase-anomaly-detection'
APP_NAME = 'app-phase-anomaly-detection'
SNAPSHOT_NAME = 'phase_anomaly_detection_buffered_snap'


def read_pv_list_from_file(pv_list_file: str) -> list[str]:
    with open(pv_list_file, 'r') as f:
        return [
            f"ca://{u:s}"
            for u in f.read().splitlines()
            if not u.startswith('#')
        ]


class K2EGHandler:
    def __init__(self,
                 pv_list: list[str],
                 snapshot_period_ms: int,
                 snapshot_handler: Callable,
                 logging_kwargs: dict = default_logging_kwargs
                 ):
        """
        This class is meant to be used as a context manager to abstract away
        the required startup and shutdown code for getting recurring buffered
        snapshots from k2eg.  It is best to run this inside a multiprocessing
        process like the K2EGProcess below.

        k2eg requires that you set an environment variable named
        K2EG_PYTHON_CONFIGURATION_PATH_FOLDER that points to a directory
        containing a file named 'lcls.ini'.
        """
        self.pv_list = pv_list
        self.snapshot_period_ms = snapshot_period_ms
        self.snapshot_handler = snapshot_handler
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'K2EGHandler'
        self.logger = None

        self.snapshot_properties = SnapshotProperties(
            snapshot_name=SNAPSHOT_NAME,
            time_window=snapshot_period_ms,  # time window of emits
            repeat_delay=0,                  # no delay between emits
            pv_uri_list=self.pv_list,
            triggered=False,                 # emit without a trigger
            type=SnapshotType.TIMED_BUFFERED,
            pv_field_filter_list=["value"]   # emit just the PV values
        )
        self.dml = k2eg.dml('lcls', APP_NAME)
        self.snapshot_is_running = False

    def __enter__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        self.logger.debug('Spinning up buffered snapshots')
        _ = self.dml.snapshot_recurring(
            self.snapshot_properties,
            handler=self.snapshot_handler,
            timeout=10,
        )
        self.snapshot_is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dml.snapshot_stop(SNAPSHOT_NAME)
        self.dml.close()
        self.snapshot_is_running = False
        self.logger.debug('shutdown snapshot production')


class K2EGProcess(CustomProcessObject):
    """
    This class creates a process for handling k2eg as a sub-process.
    It's fundamental function is to set up and break down k2eg as well
    as provide a snapshot_handler method that k2eg will call at regular
    intervals given snapshot_period_ms.

    Parameters
    ----------
    queue : Manager.Queue
        A multiprocessing manager queue on which to put the snapshots.
    pv_list : list[str]
        A list of pv names and access types to hand to k2eg.
        Example list element is 'ca://KLYS:LI20:61:PHAS_FASTBR'
    snapshot_period_ms : int
        The number of miliseconds between snapshots emitted by k2eg.
    logging_kwargs : dict
        A dictionary of arguments for the function create_worker_logger
        from the mp_logging module.
    """
    def __init__(self,
                 queue: 'Manager.Queue',
                 pv_list: list[str],
                 snapshot_period_ms: int = 1000,
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue = queue
        self.pv_list = pv_list
        self.snapshot_period_ms = snapshot_period_ms

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'K2EGProcess'
        self.logger = None

        self.k2_handler = None

        self.keep_fetching_data = False

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        if not isinstance(self.k2_handler, K2EGHandler):
            self.k2_handler = K2EGHandler(
                pv_list=self.pv_list,
                snapshot_period_ms=self.snapshot_period_ms,
                snapshot_handler=self.snapshot_handler,
                logging_kwargs=self.logging_kwargs.copy()
            )

        # put data onto the queue at regular intervals
        with self.k2_handler as k2h:
            self.logger.debug(
                "K2EGHandler.snapshot_is_running: "
                f"{k2h.snapshot_is_running}"
            )
            self.keep_fetching_data = True
            while k2h.snapshot_is_running and self.keep_fetching_data:
                # this does nothing, replace it with instrumentation
                # the sleep is to keep the status checks from being
                # nearly constant
                self.logger.debug('Waiting for data from K2EGHandler')
                time.sleep(2.0)
        self.logger.debug('Finished')
        self.queue.put(None)  # end the downstream processes

        for handler in self.logger.handlers:
            handler.close()

    def snapshot_handler(self, snapshot_name: str, snapshot: dict):
        iteration = snapshot['iteration']
        if self.logger is not None:
            self.logger.debug(
                f"Snapshot {iteration:d} "
                f"enqueued for {snapshot_name}"
            )
            self.logger.debug(snapshot)
        self.queue.put(iteration)  # processes B and C expect an integer for now


if __name__ == '__main__':
    """
    This section is for basic testing and debugging only.  Use the main
    module for deploying this code.
    """
    # basic testing
    list_of_pvs = [
        'ca://KLYS:LI20:61:PHAS_FASTBR',
        'ca://KLYS:LI20:61:AMPL'
    ]

    # test just the handler
    def snap_handler(snap_name: str, snapshot: dict):
        print(f"Snapshot from {snap_name} received: {snapshot}")

    k2_handler = K2EGHandler(
        pv_list=list_of_pvs,
        snapshot_period_ms=1000,
        snapshot_handler=snap_handler
    )

    with k2_handler as k2h:
        while True:
            time.sleep(2)
            print("Sleeping 2 seconds")

    # # test the process and handler together
    # from mp_logging import run_logger_process
    #
    # with Manager() as manager:
    #     queue_one = manager.Queue()
    #     queue_log = manager.Queue()  # for logging only
    #
    #     logging_kwargs = default_logging_kwargs = {
    #         'queue': queue_log,
    #         'logger_name': None,
    #         'log_level': 10,  # 10 is DEBUG
    #         'log_stdout': True
    #     }
    #     logger_process = Process(target=run_logger_process, kwargs=logging_kwargs)
    #     logger_process.start()
    #     main_logger = create_worker_logger(
    #         queue=logging_kwargs['queue'],
    #         logger_name='main',
    #         log_level=logging_kwargs['log_level'],
    #         log_stdout=logging_kwargs['log_stdout']
    #     )
    #
    #     main_logger.debug('starting process a')
    #     process_a = ProcessA(queue_one, pv_list=list_of_pvs, logging_kwargs=logging_kwargs.copy())
    #     main_logger.debug('process a started, engaging multiprocessing')
    #     proc_a = Process(target=process_a)
    #     proc_a.start()
    #     main_logger.debug('multiprocessing engaged')
    #
    #     logger_process.join()
    #     proc_a.join()
