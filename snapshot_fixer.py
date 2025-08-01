from data_bucketer import DataBucketer
from utils import get_timestamp_ns, get_value
from beam_check_config import SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from mp_logging import default_logging_kwargs

from collections import deque
from typing import Dict, Optional, List, Tuple
import numpy as np

class SnapshotFixer:
    """
    This class handles issues with incoming snapshots, and then buckets the data and forward-fills any missing data-points.

    Currently this class only handles a single snapshot-issue, which is when a snapshot contains > 120 data-points. This is when
    the snapshot contains data-points that should be in a following snapshot.
    (we can assume all data-points get eventually sent by k2eg, and no duplicate data-points are sent) 
    We handle this by saving any data-points outside of the current buckets time-range for later processing during their
    correct time-window.

    For more info on the bucketing and forward-filling of data, see the DataBucketer class.   
    """

    def __init__(self, pv_list: List[str], snapshot_period_ns: int, logging_kwargs: Optional[dict] = default_logging_kwargs):
        self.pv_list = pv_list
        self.time_of_first_data = 0
        self.snapshot_period_ns = snapshot_period_ns
        self.num_snapshots_processed = 0

        # store extra data-points with timestamps that belong in future snapshots
        self.temp_storage: Dict[str, deque] = {
            pv: deque() for pv in self.pv_list
        }

        self.data_bucketer = DataBucketer(SAMPLES_PER_SECOND, logging_kwargs=logging_kwargs.copy())

    def fix_snapshot(self, raw_snapshot: Dict[str, List[dict]]) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dict mapping PV -> (values, timestamps_ns) that belong in the currently being processed snapshot window.
        Early entires (entries expectred in a later snapshot) get stored in `temp_storage` for later use.

        Args:
            raw_snapshot: (Dict[str, List[dict]]): the snapshot as sent by k2eg, before any processing is applied.

        """
        if self.time_of_first_data == 0:
            min_ts = None
            for pv in self.pv_list:
                for e in raw_snapshot.get(pv, []):
                    ts = get_timestamp_ns(e)
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
            self.time_of_first_data = min_ts if min_ts is not None else 0

        start_time = self.time_of_first_data + self.num_snapshots_processed * self.snapshot_period_ns
        end_time = start_time + self.snapshot_period_ns

        fixed_snapshot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for pv in self.pv_list:
            # we can assume snapshot data is time ordered,
            # so popleft gets us oldest stored data.
            for entry in raw_snapshot.get(pv, []):
                self.temp_storage[pv].append(entry)

            in_window_values = []
            in_window_timestamps = []

            # pop entries in current timestamp-window
            while self.temp_storage[pv]:
                entry = self.temp_storage[pv][0]  # peek
                ts_ns = get_timestamp_ns(entry)

                if start_time <= ts_ns < end_time:
                    self.temp_storage[pv].popleft()
                    in_window_timestamps.append(ts_ns)
                    in_window_values.append(get_value(entry))
                else:
                    break  # data expected in future snapshot, leave in queue for later processing

            fixed_snapshot[pv] = (
                np.array(in_window_values, dtype=np.float64),
                np.array(in_window_timestamps, dtype=np.int64),
            )
        
        return fixed_snapshot

    def bucket_snapshot_data(self, fixed_snapshot: dict[str, Tuple[np.ndarray, np.ndarray]],
        prev_snapshot_val_map: dict[str, np.float64]) -> dict[str, np.ndarray]:
        """
        Use the DataBucketer class to do timestamp-based bucketing of data and forward-filling
        of missing data-points.

        Args:
            fixed_snapshot (dict[str, Tuple[np.ndarray, np.ndarray]]): This hould be the output of `fix_snapshot()`,
                a mapping of pv -> (values, timestamps) grabbed from a snapshot. (timestamps will fall into bucketing window because
                of fixing applied by `fix_snapshot()`)
    
            prev_snapshot_val_map (dict[str, np.float64]): Mapping of pv -> last value for pv in prev snapshot.

        Returns:
            dict[str, np.ndarray]: timestamp-bucketed and forward-filled mapping of pv -> data-points.

        """

        bucket_arr_start_time = self.time_of_first_data + (
            self.num_snapshots_processed * NANOSECS_IN_1_SEC
        )
        data_map_bucketed = self.data_bucketer.bucket_and_forward_fill_data(
            fixed_snapshot, prev_snapshot_val_map, bucket_arr_start_time
        )

        self.num_snapshots_processed += 1
        return data_map_bucketed