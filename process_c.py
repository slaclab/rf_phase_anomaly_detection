from copy import deepcopy
from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Optional

from inference.coad_predictor import COADPredictor
from inference.rules_based_predictor import RulesBasedPredictor
from candidate_saver import AnomalySaver


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
        queue_inst: Optional["Manager.Queue"] = None,  # instrumentation queue
        logging_kwargs: Optional[dict] = default_logging_kwargs,
    ):
        self.queue = queue
        self.queue_inst = queue_inst

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "process_c"
        self.logger = None

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        # Initialize predictors (loads models and configs, if any)
        predictors = [
            COADPredictor(write_to_pv=True, queue_inst=self.queue_inst, logger=self.logger),
            RulesBasedPredictor(logger=self.logger)
        ]

        self.logger.info(f"Predictors loaded: {[str(p) for p in predictors]}")

        self.anomaly_saver = AnomalySaver(
            directory='saved_candidates/anomalies',
            logging_kwargs=self.logging_kwargs.copy()
        )

        while True:
            if not self.queue.empty():
                r = self.queue.get()
                # self.logger.debug(f"ProcessC sees {r}")
                if r is None:  # enqueue a None to stop this process
                    break
                candidate = deepcopy(r)
                candidate["rf_pv_name"] = convert_pv_name_to_table_name(r["rf_pv_name"])

                # Run inference on the received data
                # r is a dict of the form (not showing all keys):
                # { "anomaly_timestamp": float,
                #   "rf_input": np.array of size (1, 1066),
                #   "bpm_input": np.array of size (8, 1066),
                #   "rf_pv_name": string,
                # }
                result = [
                    pred.predict(candidate=candidate)
                    for pred in predictors
                ]
                detailed_result = {str(pred): res for pred, res in zip(predictors, result)}
                self.logger.debug(f"ProcessC result: {detailed_result}")
                if any(result):
                    self.anomaly_saver.save(
                        anomaly=candidate | {'predictors': detailed_result},
                        reject=False
                    )


        # Shut down the predictor/timed dict and close the logger
        [p.shut_down() for p in predictors]
        for handler in self.logger.handlers:
            handler.close()
