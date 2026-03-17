import time
import logging
from multiprocessing import Manager, Pipe

from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger
from machine_interface.k2eg_handler import K2EGHandler
from warn_once import WarnOnceSet

from typing import Optional, Any


def get_time_from_entry(entry: dict[str, Any]) -> int:
    return entry['timeStamp']['secondsPastEpoch'] * int(1e9) + entry['timeStamp']['nanoseconds']


def sort_pv_response_by_time(pv_list: list) -> list:
    times = [get_time_from_entry(entry) for entry in pv_list]
    # see https://stackoverflow.com/a/6618543/6024187
    return [x for _, x in sorted(zip(times, pv_list), key=lambda pair: pair[0])]


def prune_alarm_data(
        pv_list: list[dict]
) -> (list[dict], str):
    """

    Parameters
    ----------
    pv_list : list[dict]
        A list of data from k2eg.

    Returns
    -------
    The pruned list and a string describing any warning-level issues with the data now that it is pruned.
    """
    pruned_list = [e for e in pv_list if e['alarm']['severity'] == 0]
    issue_string = ""
    if len(pruned_list) == 0:
        pruned_list.append(pv_list[-1])  # put the newest value back on
        issue_string = "all alarms"
    elif len(pruned_list) != len(pv_list):
        issue_string = "some alarms"
    return pruned_list, issue_string


def process_snapshot(
        snap: dict[str, Any],
        warn_once_set: WarnOnceSet,
        logger: logging.Logger
) -> dict[str, Any]:
    """
    This function processes a snapshot to make sure that the entries in each PV are
    in chronological order.

    Parameters
    ----------
    snap : dict
        A snapshot from k2eg.
    warn_once_set : WarnOnceSet
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
            if len(value) == 0:
                # this should not happen, k2eg should always return at least one value
                msg = f"PV {key:s} came from k2eg empty"
                logger.warning(msg)

            pruned_value, issue_string = prune_alarm_data(value)
            if issue_string and key not in warn_once_set:
                if issue_string == "all alarms":
                    msg = f"PV {key} was pruned to empty due to alarms, putting newest value on anyway"
                    logger.warning(msg)
                    limit = warn_once_set.keep_for_this_many_iterations
                    msg = f"Future warnings about this type of object are disabled for {limit:d} iterations."
                    logger.warning(msg)
                    warn_once_set.add(pv_name=key, reason=issue_string, iteration=iteration)
                elif issue_string == "some alarms":  # not a warning
                    msg = f"PV {key} was pruned to {len(pruned_value)} of {len(value)} values due to alarms"
                    logger.debug(msg)
                else:
                    msg = f"PV {key} was pruned for an unknown reason"
                    logger.warning(msg)
                    limit = warn_once_set.keep_for_this_many_iterations
                    msg = f"Future warnings about this type of object are disabled for {limit:d} iterations."
                    logger.warning(msg)
                    warn_once_set.add(pv_name=key, reason=issue_string, iteration=iteration)

            sorted_snapshot[key] = sort_pv_response_by_time(pruned_value)
        elif isinstance(value, int):
            sorted_snapshot[key] = value
        else:
            sorted_snapshot[key] = value
            t = type(value)
            if t not in warn_once_set:
                msg = f"Unknown type for value in snapshot {iteration:d}, key: {key:s}, type: {t}."
                logger.warning(msg)
                limit = warn_once_set.keep_for_this_many_iterations
                msg = f"Future warnings about this type of object are disabled for {limit:d} iterations."
                logger.warning(msg)
                warn_once_set.add(pv_name=t, reason="unknown type", iteration=iteration)

    removed_instances = warn_once_set.prune_old_instances(iteration)
    for ri in removed_instances:
        msg = f"Warning {ri.reason} are allowed again for PV {ri.pv_name}"
        logger.debug(msg)

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
        Example list element is 'ca://KLYS:LI20:61:PHAS_FASTCUHBR'
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
        snapshot_period_ms: Optional[int] = 1000,
        queue_inst: Optional["Manager.Queue"] = None,  # instrumentation queue
        main_pipe: Optional[Pipe] = None,
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.pv_list = pv_list
        self.snapshot_period_ms = snapshot_period_ms
        self.queue_inst = queue_inst
        self.main_pipe = main_pipe

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "K2EGProcess"
        self.logger = None

        self.k2_handler = None

        self.keep_fetching_data = False

        self.warn_once_set = None

    def __call__(self):
        self.warn_once_set = WarnOnceSet(keep_for_this_many_iterations=120)
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
                if self.main_pipe is not None and self.main_pipe.poll():
                    msg = self.main_pipe.recv()
                    if msg is None:
                        break
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
        # sorted_snapshot = snapshot

        # claudio wants us to wait for ~10 snapshots for k2eg to warm up
        n_skip = 10
        if iteration <= n_skip:
            ss = f"Skipping iteration {iteration:>2d}/{n_skip:>2d} from {snapshot_name:s} to give k2eg time to warm up"
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
    list_of_pvs = ["ca://KLYS:LI20:61:PHAS_FASTCUHBR", "ca://KLYS:LI20:61:AMPL"]

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
