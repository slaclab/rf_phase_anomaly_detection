from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional
import k2eg_spoofer

class ProcessB(CustomProcessObject):

    def __init__(self,
                 queue_one: 'Manager.Queue',
                 queue_two: 'Manager.Queue',
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue_one = queue_one
        self.queue_two = queue_two
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'process_b'
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        while True:
            if not self.queue_one.empty():
                r = self.queue_one.get()
                self.logger.debug(f"ProcessB sees {r}")
                if r is None or r % 2 == 1:  # enqueue only the odds
                    self.logger.debug(f"ProcessB enqueuing {r}")
                    self.queue_two.put(r)
                if r is None:  # enqueue a None to stop this process
                    break

        for handler in self.logger.handlers:
            handler.close()
