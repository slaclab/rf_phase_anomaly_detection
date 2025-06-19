from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional


class ProcessC(CustomProcessObject):
    def __init__(self,
                 queue: 'Manager.Queue',
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue = queue
        self.logging_kwargs = dict(logging_kwargs)
        self.logging_kwargs['logger_name'] = 'process_c'
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        while True:
            if not self.queue.empty():
                r = self.queue.get()
                #self.logger.debug(f"ProcessC sees {r}")
                if r is None:  # enqueue a None to stop this process
                    break

        for handler in self.logger.handlers:
            handler.close()
