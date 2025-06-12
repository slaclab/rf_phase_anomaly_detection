import time
from multiprocessing import Queue
from mp_logging import create_worker_logger, default_logging_kwargs
from k2eg_spoofer import K2EGSpoofer
from process import CustomProcessObject

class K2EGSpoofProcess(CustomProcessObject):
    def __init__(
        self,
        queue: 'Queue',
        pv_list: list[str],
        n_emits: int = 3,
        emit_rate_hz: int = 1,
        logging_kwargs: dict = default_logging_kwargs,
    ):
        self.queue = queue
        self.pv_list = pv_list
        self.n_emits = n_emits
        self.emit_rate_hz = emit_rate_hz
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'K2EGSpoofProcess'
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.debug("starting k2eg spoofing process")
        spoofer = K2EGSpoofer(
            pv_configs=[{'name': pv, 'rate_hz': 1.0, 'drop_rate': 0.0} for pv in self.pv_list],
            n_emits=self.n_emits,
            emit_rate_hz=self.emit_rate_hz,
        )
        for emission in spoofer():
            self.logger.debug(f"emitting: {emission}")
            self.queue.put(emission)

        self.logger.debug("k2eg spoofing finished")
        self.queue.put(None)
        for handler in self.logger.handlers:
            handler.close()
