import numpy as np
from typing import Optional


class SlidingWindowArray:
    """
    Fixed-size sliding buffer.

    Appends new data with 'my_sliding_array.put(values)', automatically dropping oldest values when full.
    Tracks current valid next-write index and supports via getting slices with `.get(start, end)`.

    Attributes:
        data (np.ndarray): underlying data-storage.
        index (int): current write position or length of valid data.
    """

    def __init__(self, buffer_len: int, dtype: np.dtype = np.float64):
        self.data = np.empty(buffer_len, dtype=dtype)
        self.buffer_len = buffer_len
        self.index = 0

    def put(self, values: np.ndarray) -> None:
        n = len(values)
        if n > self.buffer_len:
            raise ValueError(f"too many values ({n}) for buffer size {self.buffer_len}")

        if self.index + n <= self.buffer_len:
            # have enough room without shifting, just write to next open index (this only happens during initial buffer fill-up)
            self.data[self.index : self.index + n] = values
            self.index += n
        else:
            # shift left and append to the end, this should be quick on a np.arr
            shift = n
            self.data[:-shift] = self.data[shift:]
            self.data[-shift:] = values
            self.index = self.buffer_len  # remains full

    def get(self, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """
        Return slice of the buffer from start (inclusive) to end (exclusive).

        - if start arg is None, defaults to 0.
        - if end arg is None, defaults to self.index (so returns whole array from start arg index)
        - If start arg == -1 and end arg is None, returns the last value as a 1-elem array.
        """
        if start == -1 and end is None:
            return self.data[self.index - 1 : self.index]

        s = 0 if start is None else start
        e = self.index if end is None else end

        if s < 0 or e > self.index or s > e:
            raise IndexError(f"Invalid start/end indices: {s}, {e}")

        return self.data[s:e]

    def clear(self) -> None:
        self.data.fill(np.nan)
        self.index = 0

    def is_full(self) -> bool:
        return self.index == self.buffer_len

    def __len__(self) -> int:
        return self.index
