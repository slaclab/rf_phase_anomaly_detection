from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional

from inference.predict import Predict


class ProcessC(CustomProcessObject):
    def __init__(self,
                 queue: 'Manager.Queue',
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue = queue
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'process_c'
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        # Initialize predictor (loads models and configs)
        predictor = Predict()

        while True:
            if not self.queue.empty():
                r = self.queue.get()
                self.logger.debug(f"ProcessC sees {r}")
                if r is None:  # enqueue a None to stop this process
                    break

                # Run inference on the received data
                result = predictor.predict(r)
                self.logger.debug(f"ProcessC result: {result}")

                # TODO: Write result to PVs/or back to the queue?
            #     # Send result back to the queue
            #     self.queue.put(result)
            #     self.logger.debug("ProcessC put result back to queue")
            #     self.queue.task_done()
            #     self.logger.debug("ProcessC task done")
            # else:
            #     self.logger.debug("ProcessC queue is empty, waiting...")
            #     self.queue.join()
            #     self.queue.join(timeout=1)
            #     self.logger.debug("ProcessC queue join completed, checking again")

        for handler in self.logger.handlers:
            handler.close()
