import numpy as np
from typing import Optional
from mp_logging import create_worker_logger, default_logging_kwargs


class SlidingWindowArray:
    """
    Fixed-size sliding buffer.

    Appends new data with 'my_sliding_array.put(values)', automatically dropping oldest values when full.
    Tracks current valid next-write index and supports via getting slices with `.get(start, end)`.

    Attributes:
        data (np.ndarray): underlying data-storage.
        index (int): current write position or length of valid data.
    """

    def __init__(self, buffer_len: int, dtype: np.dtype = np.float64, pv_name: str = "", logging_kwargs: Optional[dict] = default_logging_kwargs):
        logging_kwargs["logger_name"] = "sliding_window_array"
        self.logger = create_worker_logger(**logging_kwargs)

        self.data = np.empty(buffer_len, dtype=dtype)
        self.buffer_len = buffer_len
        self.index = 0

        self.logger.info(f"Initialized SlidingWindowArray for pv {pv_name} with size {buffer_len}, dtype {dtype}")

    def put(self, values: np.ndarray) -> None:
        n = len(values)
        # self.logger.debug(f"Putting {n} new values into buffer")

        if n > self.buffer_len:
            self.logger.error(f"Too many values ({n}) for buffer size {self.buffer_len}")
            raise ValueError(f"too many values ({n}) for buffer size {self.buffer_len}")

        if self.index + n <= self.buffer_len:
            # have enough room without shifting, just write to next open index (this only happens during initial buffer fill-up)
            self.data[self.index : self.index + n] = values
            self.index += n
            # self.logger.debug(f"Wrote values at index {self.index - n} to {self.index}")
        else:
            # shift left and append to the end, this should be quick on a np.arr
            shift = n
            self.data[:-shift] = self.data[shift:]
            self.data[-shift:] = values
            self.index = self.buffer_len  # remains full
            # self.logger.debug(f"Buffer full; shifted left by {shift} and appended to end")

    def get(self, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """
        Return slice of the buffer from start (inclusive) to end (exclusive).

        - if start arg is None, defaults to 0.
        - if end arg is None, defaults to self.index (so returns whole array from start arg index)
        - If start arg == -1 and end arg is None, returns the last value as a 1-elem array.
        """
        if start == -1 and end is None:
            #self.logger.debug("Returning last value in buffer")
            return self.data[self.index - 1 : self.index]

        s = 0 if start is None else start
        e = self.index if end is None else end

        if s < 0 or e > self.index or s > e:
            self.logger.error(f"Invalid slice request: start={s}, end={e}, current index={self.index}")
            raise IndexError(f"Invalid start/end indices: {s}, {e}")

        # self.logger.debug(f"Returning buffer slice from {s} to {e}")
        return self.data[s:e]

    def clear(self) -> None:
        self.logger.info("Clearing buffer")
        self.data.fill(np.nan)
        self.index = 0

    def is_full(self) -> bool:
        # self.logger.debug(f"Buffer is full")
        return self.index == self.buffer_len

    def __len__(self) -> int:
        return self.index
