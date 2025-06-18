from multiprocessing import Manager
from process import CustomProcessObject
from mp_logging import default_logging_kwargs, create_worker_logger
from typing import Optional
import k2eg_spoofer
from queue import Empty
from datetime import datetime
import numpy as np
import os
import time

class ProcessB(CustomProcessObject):
    """
    ProcessB consumes k2eg snapshots from queue_one and buffers 5 minutes of data per PV at 120hz.
    it performs beam checks and candidate window searcing, and forwards output to queue_two for ProcessC.
    """
    def __init__(self,
                 queue_one: 'Manager.Queue',
                 queue_two: 'Manager.Queue',
                 pv_list: list[str],
                 logging_kwargs: Optional[dict] = default_logging_kwargs
                 ):
        self.queue_one = queue_one
        self.queue_two = queue_two

        self.pv_list = pv_list

        self.logging_kwargs = logging_kwargs
        self.logging_kwargs['logger_name'] = 'process_b'
        self.logger = None

        # holds up to 5 minutes of 120hz data (36000 points) per pv.
        self.buffer = Buffer(pv_list, 36000, logging_kwargs) # 3600 = 120hz * 60sec * 5mins

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.debug(f"running provess_b on {len(self.pv_list)} pvs")
        self.logger.debug("startin data processing loop...")

        while True:
            try:
                r = self.queue_one.get(timeout=0.05) # wait 50ms

                if r is None: # enqueuing a None should stop this process
                    break

                #need to be sure each iteration of this processing loop is <= 1 second
                # (new data comes each second from process_a, so data will pile-up if our processing takes over 1 second)
                start = time.perf_counter()

                # parse the k2eg snapshot and update buffer
                self.update_buffer(r)

                # process data in buffer and get stuff to pass to process_c and CoAD
                self.do_beam_checks()
                result = self.find_candidates()

                # placeholder: dummy data for ProcessC:
                fake_rf_input_tensor = np.random.rand(1, 1066).astype(np.float32)
                fake_bpm_input_tensor = np.random.rand(8, 1066).astype(np.float32)
                fake_pv_name = "fake_pv_name"

                fake_output_data = (fake_rf_input_tensor, fake_bpm_input_tensor, fake_pv_name)
                self.queue_two.put(fake_output_data)

                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000
                self.logger.debug(f"process_b iteration took : {elapsed_ms:.2f} ms")
                if elapsed_ms > 1000: # have to be <= 1 sec
                    self.logger.warning(f"process_b iteration is slow!! : {elapsed_ms:.2f} ms")

            except Empty:
                continue

        self.logger.debug("shutting down process_b")

        for handler in self.logger.handlers:
            handler.close()

    def update_buffer(self, snapshot):
        """
        Append the latest 120-sample PV snapshot into the buffer for each pv
        """
        for pv in self.pv_list:
            entries = snapshot.get(pv, [])

            values = np.empty(120, dtype=np.float64)
            for i, e in enumerate(entries):
                values[i] = e.get("value", np.nan)

            self.buffer.append(pv, values)
        
        # Update buffer index tracking
        if self.buffer.index != self.buffer.buffer_len:
            self.buffer.index += 120
        else:
            self.buffer_index == self.buffer_len - 120

    def do_beam_checks(self):
        # placeholder: beam condition logic here.
        return True

    def find_candidates(self):
        # placeholder: candidate determination logic here.
        return []

class Buffer:
    """
    Fixed-length buffer for storing a sliding window of 120hz float data per pv.
    By default stores 5 mins (36000 values) of past data.
    """
    def __init__(self, pv_list: list[str], buffer_len: int = 36000, logging_kwargs: Optional[dict] = default_logging_kwargs):
        self.pv_list = pv_list
        
        # max length of buffer
        self.buffer_len = buffer_len

        # default buffer length is 36000 to store 5 mins of data at 120hz.
        # we allocate the buffer initially to avoid potential memory-copies during array append operation.
        self.buffer_map = {
            pv: np.empty(buffer_len, dtype=np.float64) for pv in self.pv_list
        }
        self.index = 0  # tracks the next write index
        
        self.passes_beam_check = np.empty(buffer_len, dtype=bool) # whether at this timestamp all

        self.logging_kwargs = logging_kwargs
        self.logger = create_worker_logger(**self.logging_kwargs)
        self.logging_kwargs['logger_name'] = 'buffer'

    def append(self, key: str, values: np.ndarray):
        """
        Appends 120 new values for a given pv. If the buffer is full, old data is shifted to make room.
        """
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer")

        if len(values) != 120:
            raise ValueError(f"Expected array of length 120, got {len(values)}")

        curr_pv_arr = self.buffer_map[key]            
        idx = self.index

        if idx + 120 <= self.buffer_len:
            # have enough room without shifting, just write to next open index (this only happens during initial buffer fill-up)
            self.logger.debug(f"initial filling of buffer, current index {idx}")
            curr_pv_arr[idx:idx+120] = values
        else:
            # shift left and append to the end, this should be quick on a np.arr
            self.logger.debug(f"buffer is full, removing oldest data")
            curr_pv_arr[:-120] = curr_pv_arr[120:]
            curr_pv_arr[-120:] = values

    def get(self, key: str):
        """
        Get data from the buffer map for given pv.
        """
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer map")
        return self.buffer_map[key][:self.index]

    def clear(self, key: str):
        """
            Clear buffer contents for given pv.
        """
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer map.")
        self.buffer_map[key][:] = np.empty(self.buffer_len, dtype=np.float64)
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
            pv = pv[5:] # get rid of "ca://"
            filepath = os.path.join(directory, f"{pv.replace(':', '_')}.txt")

            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")
