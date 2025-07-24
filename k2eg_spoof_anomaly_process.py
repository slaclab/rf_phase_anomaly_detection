import time

from mp_logging import create_worker_logger, default_logging_kwargs
from process import CustomProcessObject
from beam_check_config import SAMPLES_PER_SECOND
from k2eg_anomaly_spoofer import K2EGAnomalySpoofer, constant_readings_dict

from typing import Optional

n_periods = 15
n_points = n_periods * SAMPLES_PER_SECOND

class K2EGSpoofAnomalyProcess(CustomProcessObject):
    def __init__(
            self,
            queue: "Manager.Queue",
            n_emits: Optional[int] = 0,
            emit_anomaly_every_n_iterations: Optional[int] = 6,
            logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.n_emits = n_emits
        self.emit_anomaly_every_n_iterations = emit_anomaly_every_n_iterations

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "K2EGAnomalyProcess"
        self.logger = None

        self.k2_handler = None
        self.snapshot_is_running = False

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        if self.k2_handler is None:
            self.k2_handler = K2EGAnomalySpoofer(
                self.n_emits, self.emit_anomaly_every_n_iterations
            )

        for snapshot in self.k2_handler():
            self.snapshot_handler('', snapshot)

        self.queue.put(None)  # end the downstream processes

        for handler in self.logger.handlers:
            handler.close()

    def snapshot_handler(self, snapshot_name: str, snapshot: dict):
        iteration = snapshot["iteration"]
        if self.logger is not None:
            self.logger.debug(f"Snapshot {iteration:d} enqueued for {snapshot_name}")
            # self.logger.debug(snapshot)
        self.queue.put(snapshot)


if __name__ == "__main__":
    from multiprocessing import Manager, Process

    from mp_logging import run_logger_process, create_worker_logger
    from process_b import ProcessB
    from process_c import ProcessC
    from process import ProcessManager
    from k2eg_process import K2EGProcess, read_pv_list_from_file

    with Manager() as manager:
        # create queues for inter-process communication
        queue_one = manager.Queue()
        queue_two = manager.Queue()
        queue_log = manager.Queue()  # for logging only

        # logging configuration
        logging_kwargs = default_logging_kwargs = {
            "queue": queue_log,
            "logger_name": None,    # do not log to file
            "log_level": 10,      # 10 is DEBUG
            "log_stdout": False,  # do not log to std out
        }

        logger_process = Process(target=run_logger_process, kwargs=logging_kwargs)
        logger_process.start()

        main_logger = create_worker_logger(
            queue=logging_kwargs["queue"],
            logger_name="test_spoof_anomaly",
            log_level=logging_kwargs["log_level"],
            log_stdout=logging_kwargs["log_stdout"],
        )

        k2eg_proc = K2EGAnomalyProcess(
            queue=queue_one,
            n_emits=14,
            emit_anomaly_every_n_iterations=6,
            logging_kwargs=logging_kwargs.copy()
        )

        process_objects = [
            k2eg_proc,
            ProcessB(queue_one, queue_two,
                     pv_list=list(constant_readings_dict.keys()),
                     logging_kwargs=logging_kwargs.copy()),
            ProcessC(queue_two, logging_kwargs=logging_kwargs.copy()),
        ]

        with ProcessManager(process_objects=process_objects) as pm:
            while pm.is_running:
                time.sleep(1)
                main_logger.debug("pm loop")

        queue_log.put(None)
        logger_process.join()