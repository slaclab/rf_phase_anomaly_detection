import k2eg
from k2eg.broker import SnapshotProperties, SnapshotType

from mp_logging import default_logging_kwargs, create_worker_logger

from typing import Callable


# APP_NAME appears to be important to accessing the kafka server
# the bottom APP_NAME works, but the top one does not
# someone needs to configure kafka
# APP_NAME = 'rf-phase-anomaly-detection'
APP_NAME = "app-phase-anomaly-detection"
SNAPSHOT_NAME = "phase_anomaly_detection_buffered_snap"


def convert_pvs_to_uris(pv_list: list[str]) -> list[str]:
    return [f"ca://{u:s}" for u in pv_list]


class K2EGHandler:
    def __init__(
        self,
        pv_list: list[str],
        snapshot_period_ms: int,
        snapshot_handler: Callable,
        logging_kwargs: dict = default_logging_kwargs,
    ):
        """
        This class is meant to be used as a context manager to abstract away
        the required startup and shutdown code for getting recurring buffered
        snapshots from k2eg.  It is best to run this inside a multiprocessing
        process like the K2EGProcess below.

        k2eg requires that you set an environment variable named
        K2EG_PYTHON_CONFIGURATION_PATH_FOLDER that points to a directory
        containing a file named 'lcls-ext.ini'.  Which contains the following
        with no quotes:
        '''
        [DEFAULT]
        kafka_broker_url=172.24.5.187:9094
        k2eg_cmd_topic=sdfk2eg-cmd-topic
        '''

        """
        self.pv_list = pv_list
        self.snapshot_period_ms = snapshot_period_ms
        self.snapshot_handler = snapshot_handler
        self.logging_kwargs = logging_kwargs
        self.logging_kwargs["logger_name"] = "K2EGHandler"
        self.logger = None

        self.snapshot_properties = SnapshotProperties(
            snapshot_name=SNAPSHOT_NAME,
            time_window=snapshot_period_ms,  # time window of emits
            repeat_delay=0,  # no delay between emits
            sub_push_delay_msec=50,  # have k2eg send data every 100 msec
            pv_uri_list=convert_pvs_to_uris(self.pv_list),
            triggered=False,  # emit without a trigger
            type=SnapshotType.TIMED_BUFFERED,
            pv_field_filter_list=["value", "timeStamp", "alarm"],
        )
        # self.dml = k2eg.dml("lcls-ext", APP_NAME)  # uses k2eg VM server
        self.dml = k2eg.dml("k2eg", APP_NAME)  # uses k2eg k8s server
        self.dml.snapshot_stop(self.snapshot_properties.snapshot_name)
        self.snapshot_is_running = False

    def __enter__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)
        self.logger.info("Spinning up buffered snapshots")
        try:
            _ = self.dml.snapshot_recurring(
                self.snapshot_properties,
                handler=self.snapshot_handler,
                timeout=10,
            )
        except k2eg.dml.OperationTimeout:
            self.logger.exception("Failed to start k2eg dml instance")
            raise
        else:
            self.snapshot_is_running = True
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dml.snapshot_stop(SNAPSHOT_NAME)
        self.dml.close()
        self.snapshot_is_running = False
        self.logger.info("shutdown snapshot production")