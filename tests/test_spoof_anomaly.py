import time

from mp_logging import create_worker_logger, default_logging_kwargs
from k2eg_spoofer import PVSpoofer
from process import CustomProcessObject
from beam_check_config import SAMPLES_PER_SECOND, EXP_TMIT_MIN, NANOSECS_IN_1_SEC

from typing import Optional

n_periods = 15
n_points = n_periods * SAMPLES_PER_SECOND

constant_readings_dict = {
    "BPMS:IN20:221:TMITCUHBR": 3 * EXP_TMIT_MIN,
    "BPMS:LI24:801:XBR": 2.0,
    "BPMS:LTUH:250:XBR": 1.5,
    "BPMS:LTUH:450:XBR": -1.0,
    "BPMS:DMPH:502:YBR": 1.0,
    "BPMS:DMPH:693:YBR": -3.0,
    "BPMS:LTUH:250:TMITBR": 4 * EXP_TMIT_MIN,
    "BPMS:LTUH:450:TMITBR": 4 * EXP_TMIT_MIN,
    "BPMS:DMPH:502:TMITBR": 4 * EXP_TMIT_MIN,
    "BPMS:DMPH:693:TMITBR": 4 * EXP_TMIT_MIN,
    "IOC:BSY0:MP01:PC_RATE": 8,
    "IOC:IN20:EV01:RG02_ACTRATE": 10,
    "STPR:BSYH:2:STD2_IN_A": 0,
    "KLYS:LI20:61:PHAS_FASTBR": 25.0,
    "KLYS:LI20:61:AMPL": 45.0,
}


def force_anomaly(constant_data: dict) -> dict:
    anom_data = {}
    for pv_name, reading_list in constant_data.items():
        if (
            pv_name.endswith("XBR")
            or pv_name.endswith("YBR")
            or pv_name.endswith("FASTBR")
            or pv_name.endswith("AMPL")
            or pv_name.endswith("TMITCUHBR")
            or pv_name.endswith("TMITBR")
        ):
            anom_start = len(reading_list) // 4
            anom_end = anom_start + len(reading_list) // 2
            anom_readings = []
            for i, reading in enumerate(reading_list):
                if anom_start <= i and i < anom_end:
                    reading["value"] *= 20
                anom_readings.append(reading)
            anom_data[pv_name] = anom_readings
        else:
            anom_data[pv_name] = reading_list
    return constant_data


class K2EGAnomalySpoofer:
    def __init__(self, n_emits: Optional[int] = 0, emit_anomaly_every_n_iterations: Optional[int] = 6) -> None:
        self.n_emits = n_emits
        self.emit_anomaly_every_n_iterations = emit_anomaly_every_n_iterations
        self.pv_configs = [{"name": name, "rate_hz": 120, "drop_rate": 0.0} for name in constant_readings_dict.keys()]
        self.emit_rate_hz = 1
        self.emit_period = 1 / self.emit_rate_hz

        self.pv_spoofers = {pvc["name"]: PVSpoofer(pvc) for pvc in self.pv_configs}

    def __call__(self):
        iteration = 0
        if self.n_emits <= 0:
            i = int(-1e10)
        else:
            i = 0

        start_time = time.time_ns()
        while i < self.n_emits:
            # do some stuff
            pvs = {
                k: v.emit_readings(
                    start_time,
                    start_time + self.emit_period * NANOSECS_IN_1_SEC,
                    constant_reading=constant_readings_dict[k],
                )
                for k, v in self.pv_spoofers.items()
            }
            if iteration % self.emit_anomaly_every_n_iterations == 0:
                pvs = force_anomaly(pvs)
            emission = {"iteration": iteration, "timestamp": int(time.time() * 1000)}
            emission.update(pvs)
            iteration += 1
            i += 1
            start_time += NANOSECS_IN_1_SEC
            yield emission
            time.sleep(self.emit_period)


class K2EGAnomalyProcess(CustomProcessObject):
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
            self.k2_handler = K2EGAnomalySpoofer(self.n_emits, self.emit_anomaly_every_n_iterations)

        for snapshot in self.k2_handler():
            self.snapshot_handler("", snapshot)

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

    with Manager() as manager:
        # create queues for inter-process communication
        queue_one = manager.Queue()
        queue_two = manager.Queue()
        queue_log = manager.Queue()  # for logging only

        # logging configuration
        logging_kwargs = default_logging_kwargs = {
            "queue": queue_log,
            "logger_name": None,  # do not log to file
            "log_level": 10,  # 10 is DEBUG
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
            queue=queue_one, n_emits=14, emit_anomaly_every_n_iterations=6, logging_kwargs=logging_kwargs.copy()
        )

        process_objects = [
            k2eg_proc,
            ProcessB(
                queue_one, queue_two, pv_list=list(constant_readings_dict.keys()), logging_kwargs=logging_kwargs.copy()
            ),
            ProcessC(queue_two, logging_kwargs=logging_kwargs.copy()),
        ]

        with ProcessManager(process_objects=process_objects) as pm:
            while pm.is_running:
                time.sleep(1)
                main_logger.debug("pm loop")

        queue_log.put(None)
        logger_process.join()
