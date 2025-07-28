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

    def get_timestamp_ns(self, entry: dict) -> int:
        ts = entry.get("timeStamp", {})
        seconds = ts.get("secondsPastEpoch", 0)
        nanos = ts.get("nanoseconds", 0)
        return int(seconds * 1e9 + nanos)

    def get_value(self, entry: dict) -> float:
        if "value" in entry:
            return entry["value"]
        elif "index" in entry:
            return entry["index"]
        else:
            return float("nan")

    def fix_snapshot(
        self, raw_snapshot: Dict[str, List[dict]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a dict mapping PV -> (values, timestamps_ns) that belong to this snapshot window.
        Early entries get stored in `temp_storage` for later use.
        """


        if self.initial_time_ns == 0:
            min_ts = None
            for pv in self.pv_list:
                for e in raw_snapshot.get(pv, []):
                    ts = self.get_timestamp_ns(e)
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
            self.initial_time_ns = min_ts if min_ts is not None else 0

        start_time = self.initial_time_ns + self.num_snapshots_processed * self.snapshot_period_ns
        end_time = start_time + self.snapshot_period_ns
        self.num_snapshots_processed += 1

        fixed_snapshot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for pv in self.pv_list:
            current_entries = raw_snapshot.get(pv, [])
            temp_entries = self.temp_storage[pv]

            skip_fix = not temp_entries and all(
                start_time <= self.get_timestamp_ns(e) < end_time for e in current_entries
            )

            if skip_fix:
                values = np.array([self.get_value(e) for e in current_entries], dtype=np.float64)
                timestamps = np.array([self.get_timestamp_ns(e) for e in current_entries], dtype=int)
                fixed_snapshot[pv] = (values, timestamps)
                continue

            full_entries = list(temp_entries) + current_entries

            self.temp_storage[pv].clear()

            in_window_values = []
            in_window_timestamps = []

            for entry in full_entries:
                ts_ns = self.get_timestamp_ns(entry)
                if start_time <= ts_ns < end_time:
                    in_window_values.append(self.get_value(entry))
                    in_window_timestamps.append(ts_ns)
                else:
                    self.temp_storage[pv].append(entry)

            fixed_snapshot[pv] = (
                np.array(in_window_values, dtype=np.float64),
                np.array(in_window_timestamps, dtype=np.int64),
            )

        return fixed_snapshot
