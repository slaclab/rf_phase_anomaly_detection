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
                # r should have the structure (rf_input_tensor, bpm_input_tensor, rf_station)
                # where rf_input_tensor and bpm_input_tensor are tensors of size (1, 1066)
                # and (8, 1066) respectively, and rf_station is a string representing the PV name
                result = predictor.predict(r, write_to_pv=True)
                self.logger.debug(f"ProcessC result: {result}")

        for handler in self.logger.handlers:
            handler.close()
