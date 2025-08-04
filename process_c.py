from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional

from inference.predict import Predict


def convert_pv_name_to_table_name(name: str) -> str:
    """
    Converts name from something that looks like KLYS:LI20:61:PHAS_FASTBR to
    something that looks like klys_li20_61.  The former is used by K2EG and
    the latter is used by the NTTable for the GUI.
    """
    if len(name.split(":")) == 4:
        return "_".join(name.split(":")[:-1]).lower()
    else:
        raise NotImplementedError(
            f"process_c expected a name that looks like KLYS:LI20:61:PHAS_FASTBR, but received {name} instead"
        )


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
                # r is a dict of the form (not showing all keys):
                # { "anomaly_timestamp": float,
                #   "rf_input": np.array of size (1, 1066),
                #   "bpm_input": np.array of size (8, 1066),
                #   "rf_pv_name": string,
                # }
                result = predictor.predict(
                    **{
                        "rf_input": r["rf_input"],
                        "bpm_input": r["bpm_input"],
                        "rf_pv_name": convert_pv_name_to_table_name(r["rf_pv_name"]),
                        "anomaly_timestamp": r["anomaly_timestamp"],
                    }
                )
                self.logger.debug(f"ProcessC result: {result}")

        # Shut down the predictor/timed dict and close the logger
        predictor.shut_down()
        for handler in self.logger.handlers:
            handler.close()
