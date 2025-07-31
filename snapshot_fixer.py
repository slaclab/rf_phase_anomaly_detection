from utils import get_timestamp_ns, get_value

from collections import deque
from typing import Dict, List, Tuple
import numpy as np

class SnapshotFixer:
    def __init__(self, pv_list: List[str], snapshot_period_ns: int):
        self.pv_list = pv_list
        self.initial_time_ns = 0
        self.snapshot_period_ns = snapshot_period_ns
        self.num_snapshots_processed = 0

        # store extra data-points with timestamps that belong in future snapshots
        self.temp_storage: Dict[str, deque] = {
            pv: deque() for pv in self.pv_list
        }

    def fix_snapshot(
        self, raw_snapshot: Dict[str, List[dict]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dict mapping PV -> (values, timestamps_ns) that belong in the currently being processed snapshot window.
        Early entires (entries expectred in a later snapshot) get stored in `temp_storage` for later use.
        """

        if self.initial_time_ns == 0:
            min_ts = None
            for pv in self.pv_list:
                for e in raw_snapshot.get(pv, []):
                    ts = get_timestamp_ns(e)
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
            self.initial_time_ns = min_ts if min_ts is not None else 0

        start_time = self.initial_time_ns + self.num_snapshots_processed * self.snapshot_period_ns
        end_time = start_time + self.snapshot_period_ns
        self.num_snapshots_processed += 1

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