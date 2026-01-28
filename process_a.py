import time
import logging
from multiprocessing import Manager

from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger
from k2eg_interface.k2eg_handler import K2EGHandler

from typing import Optional, Any


def sort_pv_response_by_time(pv_list: list) -> list:
    times = [
        entry['timeStamp']['secondsPastEpoch'] * int(1e9) + entry['timeStamp']['nanoseconds']
        for entry in pv_list
    ]
    # see https://stackoverflow.com/a/6618543/6024187
    return [x for _, x in sorted(zip(times, pv_list))]

def process_snapshot(
        snap: dict[str, Any],
        warn_once_set: set,
        logger: logging.Logger
) -> dict[str, Any]:
    """
    This function processes a snapshot to make sure that the entries in each PV are
    in chronological order.

    Parameters
    ----------
    snap : dict
        A snapshot from k2eg.
    warn_once_set : set
        A set for holding the types of entries that have already warned about.
    logger : logging.Logger
        A logger object.

    Returns
    -------
    The processed snapshot.
    """
    iteration = snap["iteration"]
    t0 = time.time()
    sorted_snapshot = {}
    for key, value in snap.items():
        if isinstance(value, list):
            sorted_snapshot[key] = sort_pv_response_by_time(value)
        elif isinstance(value, int):
            sorted_snapshot[key] = value
        else:
            sorted_snapshot[key] = value
            t = type(value)
            if t not in warn_once_set:
                msg = f"Unknown type for value in snapshot {iteration:d}, key: {key:s}, type: {t}. "
                msg += "Future warnings about this type of object are disabled."
                logger.warning(msg)
                warn_once_set.add(t)
    if iteration % 20 == 0:
        dt = time.time() - t0
        logger.debug(f"snapshot {iteration:d} sorting time: {1000 * dt:.1f} ms")
    return sorted_snapshot



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
        The number of milliseconds between snapshots emitted by k2eg.
    logging_kwargs : dict
        A dictionary of arguments for the function create_worker_logger
        from the mp_logging module.
    """

    def __init__(
        self,
        queue: "Manager.Queue",
        pv_list: list[str],
        snapshot_period_ms: int = 1000,
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.pv_list = pv_list
        self.snapshot_period_ms = snapshot_period_ms

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "K2EGProcess"
        self.logger = None

        self.k2_handler = None

        self.keep_fetching_data = False

        self.warn_once_set = set()

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        if not isinstance(self.k2_handler, K2EGHandler):
            self.k2_handler = K2EGHandler(
                pv_list=self.pv_list,
                snapshot_period_ms=self.snapshot_period_ms,
                snapshot_handler=self.snapshot_handler,
                logging_kwargs=self.logging_kwargs.copy(),
            )

        # put data onto the queue at regular intervals
        with self.k2_handler as k2h:
            self.logger.info(f"K2EGHandler.snapshot_is_running: {k2h.snapshot_is_running}")
            self.keep_fetching_data = True
            while k2h.snapshot_is_running and self.keep_fetching_data:
                # this does nothing, replace it with instrumentation
                # the sleep is to keep the status checks from being
                # nearly constant
                self.logger.debug("Waiting for data from K2EGHandler")
                time.sleep(0.2)
        self.logger.debug("Finished")
        self.queue.put(None)  # end the downstream processes

        for handler in self.logger.handlers:
            handler.close()

    def snapshot_handler(self, snapshot_name: str, snapshot: dict):
        """
        This function is passed to k2eg handlers to tell them what to do with a snapshot.
        The responses from the PVs are usually, but not always, in chronological order.
        Sort them here.

        Parameters
        ----------
        snapshot_name : str
            The name of the snapshot.
        snapshot : dict
            Keys are strings, values can be anything.
            ints are 'iteration', 'header_timestamp', 'tail_timestamp', and 'timestamp'
            lists are the response from the PVs.
        """
        iteration = snapshot["iteration"]

        sorted_snapshot = process_snapshot(snapshot, self.warn_once_set, self.logger)

        # claudio wants us to wait for ~10 snapshots for k2eg to warm up
        n_skip = 10
        if iteration <= n_skip:
            ss = f"Skipping iteration {iteration}/{n_skip} from {snapshot_name:s} to give k2eg time to warm up"
            self.logger.info(ss)
        else:
            if self.logger is not None:
                self.logger.debug(f"Snapshot {iteration:d} enqueued for {snapshot_name:s}")
                # self.logger.debug(sorted_snapshot)
            self.queue.put(sorted_snapshot)


if __name__ == "__main__":
    """
    This section is for basic testing and debugging only.  Use the main
    module for deploying this code.
    """
    # basic testing
    list_of_pvs = ["ca://KLYS:LI20:61:PHAS_FASTBR", "ca://KLYS:LI20:61:AMPL"]

    # test just the handler
    def snap_handler(snap_name: str, snapshot: dict):
        print(f"Snapshot from {snap_name} received: {snapshot}")

    k2_handler = K2EGHandler(pv_list=list_of_pvs, snapshot_period_ms=1000, snapshot_handler=snap_handler)

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
