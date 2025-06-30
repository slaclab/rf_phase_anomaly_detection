from typing import Optional
import numpy as np
import logging

from mp_logging import create_worker_logger

SAMPLES_PER_SECOND = 120 # hz
BUFFER_DURATION_SEC = 60 * 5  # 5 mins
BUFFER_LENGTH = SAMPLES_PER_SECOND * BUFFER_DURATION_SEC

class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """
    def __init__(self, pv_list: list[str], buffer_len: int, logger: logging.Logger):
        self.pv_list = pv_list
        
        # max length of buffer
        self.buffer_len = buffer_len

        self.index = 0 # tracks the next write index
        # default buffer length is 36000 to store 5 mins of data at 120hz.
        # we allocate the buffer initially to avoid potential memory-copies during array append operation.
        self.buffer_map = {
            pv: np.empty(buffer_len, dtype=np.float64) for pv in self.pv_list
        }
        # just store the timestamp data from the first pv we read from the snapshot,
        # and assume the other pv's data is timed the same.
        self.pv_timestamps = np.empty(buffer_len, dtype=np.float64)
        self.passes_beam_checks = np.empty(buffer_len, dtype=bool)

        self.logger = logger

    def append(self, key: str, num_new_data_points: int, values: np.ndarray, timestamps: Optional[np.ndarray]):
        """
        Appends 'num_new_data_points' of data new values into the buffer mapping for a given pv. If the buffer is full, old data is shifted to make room.
        Also appends timestamp data if 'timestamps' arg is not None.
        """
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer")

        if len(values) != SAMPLES_PER_SECOND:
            raise ValueError(f"Expected array of length SAMPLES_PER_SECOND, got {len(values)}")

        curr_pv_arr = self.buffer_map[key]            
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

    def get(self, key: str):
        """
        Get data from the buffer map for given pv.
        """
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer map")
        return self.buffer_map[key][:self.index]

    def clear(self):
        """
        Clear buffer contents for all PVs.
        """
        for key in self.buffer_map:
            self.buffer_map[key][:] = np.empty(self.buffer_len, dtype=np.float64) # ':' will hopefully modify in place
        self.pv_timestamps[:] = np.empty(self.buffer_len, dtype=np.float64)
        self.passes_beam_checks[:] = np.empty(self.buffer_len, dtype=bool)
        self.index = 0

    def dump_to_human_readable(self, directory: str = "buffer_dump_txt"):
        """
        Debug util: dump each PV's data to a separate text file, 1 value per line.
        
        Call like this:
            dir_name = f"buffer_txt_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.debug(f"dump dir: {dir_name}")
            self.buffer.dump_to_human_readable(directory=dir_name)
        """
        os.makedirs(directory, exist_ok=True)
        
        for pv in self.pv_list:
            valid_data = self.buffer_map[pv][:self.index]
            filename = pv.lstrip("ca://").replace(':', '_') + ".txt"
            filepath = os.path.join(directory, filename)
            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")
