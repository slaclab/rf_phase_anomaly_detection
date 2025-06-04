import time
from multiprocessing import Process, Manager
import k2eg
from k2eg.broker import SnapshotProperties, SnapshotType
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional


# APP_NAME appears to be import to accessing the kafka server
# the bottom APP_NAME works, but the top one does not
# someone needs to configure kafka, I guess?
# APP_NAME = 'rf-phase-anomaly-detection'
APP_NAME = 'app-phase-anomaly-detection'
SNAPSHOT_NAME = 'phase_anomaly_detection_buffered_snap'


class K2EGHandler:
    def __init__(self,
                 pv_list: list[str],
                 snapshot_handler
                 ):
        self.pv_list = pv_list
        self.snapshot_handler = snapshot_handler
        self.snapshot_properties = SnapshotProperties(
            snapshot_name=SNAPSHOT_NAME,
            time_window=1000,
            repeat_delay=0,
            pv_uri_list=self.pv_list,
            triggered=False,
            type=SnapshotType.TIMED_BUFFERED,
            pv_field_filter_list=["value"]
        )
        self.dml = k2eg.dml('lcls', APP_NAME)
        self.snapshot_is_running = False

    def __enter__(self):
        # requires environment variable K2EG_PYTHON_CONFIGURATION_PATH_FOLDER
        # to be set to the location of a file named lcls.ini
        print('K2EGHandler: Spinning up buffered snapshots')
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
        print('K2EGHandler: shutdown snapshot production')


class ProcessA(CustomProcessObject):
    def __init__(self,
                 queue: 'Manager.Queue',
                 pv_list: list[str],
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue = queue
        self.pv_list = pv_list

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'k2egHandler'
        self.logger = None

        self.k2_handler = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        if not isinstance(self.k2_handler, K2EGHandler):
            self.k2_handler = K2EGHandler(
                pv_list=self.pv_list,
                snapshot_handler=self.snapshot_handler
            )

        # put data onto the queue at regular intervals
        with self.k2_handler as k2h:
            self.logger.debug(f"ProcessA K2EGHandler.snapshot_is_running: {k2h.snapshot_is_running}")
            while k2h.snapshot_is_running:
                self.logger.debug('ProcessA is waiting for data from K2EGHandler')
                time.sleep(2.0)
        self.logger.debug('ProcessA finished')
        self.queue.put(None)  # end the downstream processes

        for handler in self.logger.handlers:
            handler.close()

    def snapshot_handler(self, snapshot_name: str, snapshot: dict):
        if self.logger is not None:
            self.logger.debug(f"Snapshot enqueued for {snapshot_name}")
        self.queue.put((snapshot_name, snapshot))


if __name__ == '__main__':
    # basic testing
    list_of_pvs = ['ca://KLYS:LI20:61:PHAS_FASTBR', 'ca://KLYS:LI20:61:AMPL']

    # test just the handler
    def snap_handler(snap_name: str, snapshot: dict):
        print(f"Snapshot from {snap_name} received: {snapshot}")

    k2_handler = K2EGHandler(pv_list=list_of_pvs, snapshot_handler=snap_handler)

    with k2_handler as k2h:
        while True:
            time.sleep(2)
            print("Sleeping 2 seconds")

    # # test the process and handler together
    # from multiprocessing import Manager
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
