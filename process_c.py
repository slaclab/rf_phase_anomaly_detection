from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional

from inference.predict import Predict


class ProcessC(CustomProcessObject):
    def __init__(
        self,
        queue: "Manager.Queue",
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "process_c"
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        # Initialize predictor (loads models and configs)
        predictor = Predict(write_to_pv=True, logger=self.logger)

        # TEMPORARY: Silence lume-model out of range warnings
        predictor.networks[0].model.input_validation_config = {
            n: "none" for n in predictor.networks[0].model.input_names
        }
        predictor.networks[1].model.input_validation_config = {
            n: "none" for n in predictor.networks[1].model.input_names
        }

        while True:
            if not self.queue.empty():
                r = self.queue.get()
                # self.logger.debug(f"ProcessC sees {r}")
                if r is None:  # enqueue a None to stop this process
                    break

                # Run inference on the received data
                # r is a dict of the form
                # { "timestamp": float,
                #   "rf_input": np.array of size (1, 1066),
                #   "bpm_input": np.array of size (8, 1066),
                #   "pv_name": string,
                # }
                result = predictor.predict(**r)
                self.logger.debug(f"ProcessC result: {result}")

        # Shut down the predictor/timed dict and close the logger
        predictor.shut_down()
        for handler in self.logger.handlers:
            handler.close()
