from data_bucketer import DataBucketer
from beam_check_config import SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from mp_logging import create_worker_logger, default_logging_kwargs

from collections import deque
from typing import Dict, Optional, List, Tuple
import numpy as np

class SnapshotFixer:
    """
    This class handles issues with incoming snapshots, and then buckets the data and forward-fills any missing data-points.

    Currently this class only handles a single snapshot-issue, which is when a snapshot contains > 120 data-points. This is when
    the snapshot contains data-points that should be in a following snapshot.

    We handle this case by saving any data-points outside of the current buckets time-range for later processing during their
    correct time-window.

    We can assume the following about data from k2eg:
        - all data-points from pv get eventually sent (in our case, assume we will get all 120 data-points for each second in *some* snapshot eventually)
        - data-points are sent in order (oldest to newest based on timestamp)
        - no duplicate data-points are sent
    
    For more info on the bucketing and forward-filling of data, see the DataBucketer class.   
    """

    def __init__(self, pv_list: List[str], logging_kwargs: Optional[dict] = default_logging_kwargs):
        logging_kwargs["logger_name"] = "snapshot_fixer"
        self.logger = create_worker_logger(**logging_kwargs)

        self.pv_list = pv_list
        self.time_of_first_data = 0

        # store extra data-points with timestamps that belong in future snapshots
        self.temp_storage: Dict[str, deque] = {
            pv: deque() for pv in self.pv_list
        }

        self.data_bucketer = DataBucketer(SAMPLES_PER_SECOND, logging_kwargs=logging_kwargs.copy())

    def fix_snapshot(self, raw_snapshot: Dict[str, List[dict]], start_time: int, end_time: int) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dict mapping PV -> (values, timestamps_ns) that belong in the currently being processed snapshot window.
        Early entires (entries expectred in a later snapshot) get stored in `temp_storage` for later use.

        Args:
            raw_snapshot (Dict[str, List[dict]]): the snapshot as sent by k2eg, before any processing is applied.

            start_time (int): starting time of first bucket for this snapshot (usually = (time_of_first_data + (self.num_snapshots_processed * self.snapshot_period_ns))
            
            end_time (int): ending time of last bucket for thie snapshot (usually = start_time + self.snapshot_period_ns)
        """

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

                if ts_ns < start_time:
                # late data (data that belongs in previous snapshot) should not get sent, log a warning so we will know if it somehow happens
                    self.logger.warning(f"Snapshot {self.num_snapshots_processed} had a late data-point!")
                    self.temp_storage[pv].popleft()  # just throw this data-point away for now (handle later if recurring issue)
                elif ts_ns < end_time:
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
        prev_snapshot_val_map: dict[str, np.float64], start_time: int, end_time: int) -> dict[str, np.ndarray]:
        """
        Use the DataBucketer class to do timestamp-based bucketing of data and forward-filling
        of missing data-points.

        Args:
            fixed_snapshot (dict[str, Tuple[np.ndarray, np.ndarray]]): This hould be the output of `fix_snapshot()`,
                a mapping of pv -> (values, timestamps) grabbed from a snapshot. (timestamps will fall into bucketing window because
                of fixing applied by `fix_snapshot()`)
    
            prev_snapshot_val_map (dict[str, np.float64]): Mapping of pv -> last value for pv in prev snapshot.

            start_time (int): starting time of first bucket for this snapshot (usually = (time_of_first_data + (self.num_snapshots_processed * self.snapshot_period_ns))

            end_time (int): ending time of last bucket for thie snapshot (usually = start_time + self.snapshot_period_ns)

        Returns:
            dict[str, np.ndarray]: timestamp-bucketed and forward-filled mapping of pv -> data-points.
        """
        self.logger.debug("Bucketing and forward-filling data...")
        data_map_bucketed = self.data_bucketer.bucket_and_forward_fill_data(
            fixed_snapshot, prev_snapshot_val_map, start_time
        )

        return data_map_bucketed

# snapshot-event processing helper functions
def get_timestamp_ns(entry: dict) -> int:
    """
    Extract the timestamp in nanoseconds from a PV data-point from a snapshot entry.

    Parameters:
        entry (dict): A dictionary for a givin PV in a snapshot.
        (can get this by doing: `snapshot.get(pv, [])`)

    Returns:
        int: The timestamp of a PV data-point in nanoseconds
    """
    ts = entry.get("timeStamp", {})
    seconds = ts.get("secondsPastEpoch", 0)
    nanos = ts.get("nanoseconds", 0)
    return int(seconds * NANOSECS_IN_1_SEC + nanos)

def get_value(entry: dict) -> float:
    """
    Extract the value of a singe PV data-point from a snapshot entry.

    Parameters:
        entry (dict): A dictionary for a givin PV in a snapshot.
        (can get this by doing: `snapshot.get(pv, [])`)

    Returns:
        float: The PV data-point, or NaN if not valid in the snapshot entry.
    """
    if "value" in entry:
        return entry["value"]
    elif "index" in entry:
        return entry["index"]
    else:
        return float("nan")