from typing import Optional
import numpy as np
import logging

from mp_logging import create_worker_logger
from beam_check import do_beam_checks
from beam_check_config import (
    BEAM_RATE_PV,
    BEAM_SPLIT_PV,
    IN_TMIT_PV,
    STOPPER_PV,
    BEAM_RATE_TABLE,
    BEAM_SPLIT_TABLE_HXR,
    MIN_VIOLATION_DUR,
    EXP_TMIT_FREQ,
    ALLOWED_TMIT_DIFF,
    EXP_TMIT_MIN,
)
from sliding_window import SlidingWindowArray

SAMPLES_PER_SECOND = 120 # hz
BUFFER_DURATION_SEC = 60 * 5  # 5 mins
BUFFER_LENGTH = SAMPLES_PER_SECOND * BUFFER_DURATION_SEC
MIN_WINDOW_LEN = SAMPLES_PER_SECOND * 90

class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """
    def __init__(self, pv_list: list[str], buffer_len: int, logger: logging.Logger) -> None:
        self.pv_list = pv_list
        
        # max length of buffer
        self.buffer_len = buffer_len

        self.index = 0 # tracks the next write index
        # default buffer length is 36000 to store 5 mins of data at 120hz.
        # we allocate the buffer initially to avoid potential memory-copies during array append operation.
        self.data_map = {
            pv: SlidingWindowArray(buffer_len, dtype=np.float64) for pv in self.pv_list
        }
        # just store the timestamp data from the first pv we read from the snapshot,
        # and assume the other pv's data is timed the same.
        self.data_map["pv_timestamps"] = SlidingWindowArray(buffer_len, dtype=np.float64)
        self.data_map["beam_checks"] = SlidingWindowArray(buffer_len, dtype=bool)

        self.valid_windows = np.empty(0, dtype=object) # will hold tuples of (window_start_index, window_end_index)

        self.logger = logger

    def update(self, snapshot: dict[str, list[dict]]) -> None:
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv
        """
        for i, pv in enumerate(self.pv_list):
            entries = snapshot.get(pv, [])

            values = np.empty(SAMPLES_PER_SECOND, dtype=np.float64)
            for j, e in enumerate(entries):
                values[j] = e.get("value", np.nan)

            self.data_map[pv].put(values)

            times = None
            # update buffer's timestamps arr with the first pv's times,
            # and assume the other pv have same timing.
            if i == 0:
                timestamps = np.empty(SAMPLES_PER_SECOND, dtype=np.float64)
                for j, e in enumerate(entries):
                    ts = e.get("timeStamp", {})
                    seconds = ts.get("secondsPastEpoch", 0)
                    nanos = ts.get("nanoseconds", 0)
                    timestamps[j] = seconds + nanos * 1e-9

                self.data_map["pv_timestamps"].put(timestamps)

        self_index = self.data_map[self.pv_list[0]].index  # use first pv as index reference

        do_beam_checks(
            stopper_pv_data=self.data_map[STOPPER_PV],
            beam_rate_pv_data=self.data_map[BEAM_RATE_PV],
            beam_split_pv_data=self.data_map[BEAM_SPLIT_PV],
            in_tmit_pv_data=self.data_map[IN_TMIT_PV],
            starting_index=self.index,
            num_samples_to_check=SAMPLES_PER_SECOND,
            beam_checks=self.data_map["beam_checks"]
        )

        self.update_violation_window_list()

        # Update buffer index tracking
        if self.index != self.buffer_len:
            self.index += SAMPLES_PER_SECOND
        else:
            self.index == self.buffer_len - SAMPLES_PER_SECOND
        
    def append(self, key: str, num_new_data_points: int, values: np.ndarray, timestamps: Optional[np.ndarray]) -> None:
        """
        Appends 'num_new_data_points' of data new values into the buffer mapping for a given pv. If the buffer is full, old data is shifted to make room.
        Also appends timestamp data if 'timestamps' arg is not None.
        """
        if key not in self.data_map:
            raise KeyError(f"key '{key}' not found in buffer")

        if len(values) != SAMPLES_PER_SECOND:
            raise ValueError(f"Expected array of length SAMPLES_PER_SECOND, got {len(values)}")

        curr_pv_arr = self.data_map[key]            
        idx = self.index

        if idx + num_new_data_points <= self.buffer_len:
            # have enough room without shifting, just write to next open index (this only happens during initial buffer fill-up)
            #self.logger.debug(f"initial filling of buffer, current index {idx}")
            curr_pv_arr[idx:idx+num_new_data_points] = values
            if timestamps is not None:
                self.pv_timestamps[idx:idx+num_new_data_points] = timestamps
        else:
            # shift left and append to the end, this should be quick on a np.arr
            #self.logger.debug(f"buffer is full, removing oldest data")
            curr_pv_arr[:-num_new_data_points] = curr_pv_arr[num_new_data_points:]
            curr_pv_arr[-num_new_data_points:] = values
            if timestamps is not None:
                self.pv_timestamps[:-num_new_data_points] = self.pv_timestamps[num_new_data_points:]
                self.pv_timestamps[-num_new_data_points:] = timestamps

    def update_violation_window_list(self) -> list[tuple[int, int]]:
        """
        Search buffer's passes_beam_checks array to find contiguous regions of failing beam check of >= TEMP_VIOLATION_LENGTH seconds.
        Returns list of (start_index, end_index) tuples, where end_index is the first index where the beam-checks start to pass again.
        """
        windows = []
        start = None

        beam_check_arr = self.data_map["beam_checks"]
        for i, val in enumerate(beam_check_arr.get()): # .get() with no args returns the entire arr
            if val is False:
                if start is None:
                    start = i  # begin a new potential failure window
            else:
                if start is not None:
                    window_len = i - start
                    if window_len >= MIN_WINDOW_LEN:
                        windows.append((start, i))
                    start = None  # end current window

        # handle trailing window that runs to end of buffer
        if start is not None:
            window_len = len(self.passes_beam_checks) - start
            if window_len >= MIN_WINDOW_LEN:
                windows.append((start, len(self.passes_beam_checks)))

        return windows

    def find_candidates(self) -> list[dict]:
        # placeholder: candidate determination logic here.
        return []

    def get(self, key: str, start_index: Optional[int] = None, end_index: Optional[int] = None) -> np.ndarray:
        """
        Get data from the buffer map for given pv.
        Will return all the valid data for specified pv in buffer if start_index and end_index are None,
        else will return the data in the specified range. (or an empty array if the specified range is not valid)
        """
        if key not in self.data_map:
            raise KeyError(f"key '{key}' not found in buffer map")

        if start_index < 0 or start_index > self.index or end_index > self.index:
            raise IndexError(f"start and end indicies not valid in buffer: {start_index}, {end_index}")
            return []

        s = start_index if start_index   is not None else 0
        e = end_index if end_index is not None else self.index
        return self.data_map[key][s:e]

    def clear(self) -> None:
        """
        Clear buffer contents for all PVs.
        """
        for key in self.data_map:
            self.data_map[key][:] = np.empty(self.buffer_len, dtype=np.float64) # ':' will hopefully modify in place
        self.pv_timestamps[:] = np.empty(self.buffer_len, dtype=np.float64)
        self.passes_beam_checks[:] = np.empty(self.buffer_len, dtype=bool)
        self.index = 0

    def dump_to_human_readable(self, directory: str = "buffer_dump_txt") -> None:
        """
        Debug util: dump each PV's data to a separate text file, 1 value per line.
        
        Call like this:
            dir_name = f"buffer_txt_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.debug(f"dump dir: {dir_name}")
            self.buffer.dump_to_human_readable(directory=dir_name)
        """
        os.makedirs(directory, exist_ok=True)
        
        for pv in self.pv_list:
            valid_data = self.data_map[pv][:self.index]
            filename = pv.lstrip("ca://").replace(':', '_') + ".txt"
            filepath = os.path.join(directory, filename)
            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")
