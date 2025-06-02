import time
from multiprocessing import Process, Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional


class ProcessA(CustomProcessObject):
    def __init__(self,
                 queue: 'Manager.Queue',
                 n: int = 10,
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue = queue
        self.n = n
        self.process = Process(target=self)
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'process_a'
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        # put data onto the queue at regular intervals
        i = 0
        while i < self.n:
            # print(f"ProcessA enqueuing {i:d}")
            self.logger.debug(f"ProcessA enqueuing {i:d}")
            self.queue.put(i)
            i += 1
            time.sleep(0.5)
        self.queue.put(None)  # end the downstream processes

        for handler in self.logger.handlers:
            handler.close()
