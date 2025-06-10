from multiprocessing import Manager, Process
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional

import time

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

        self.buffer = Manager().list()
        self.buffer_loader_process = None

    def buffer_loader(self):
        self.logger.debug("Buffer-loader started")
        while True:
            #self.logger.debug("buffer-loader loop iter!!")
            r = self.queue_one.get() # waits on get unless at timeout arg
            logger.debug(f"Buffer-loader got: {r}")
            # don't append here if r's timestamp is older than 5 mins 
            self.buffer.append(r)
            if r is None:
                break
        logger.debug("Buffer-loader exiting")
        for handler in logger.handlers:
            handler.close()

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
            
        self.loader_process = Process(target=self.buffer_loader)
        self.loader_process.start()
        
        self.logger.debug("ProcessB started")
        
        while True:
            #self.logger.debug("process-b loop iter!!")
            # wait for buffer_loader to load some stuff
            if len(self.buffer) == 0:
                time.sleep(0.01)
                continue

            r = self.buffer.pop(0)
            self.logger.debug(f"ProcessB sees {r}")
            # process PVs for beam checks, determining window, etc
            self.queue_two.put(r)
            if r is None: # enqueue a None to stop this process
                break

        self.logger.debug("ProcessB exiting")
        self.loader_process.join()
        for handler in self.logger.handlers:
            handler.close()
