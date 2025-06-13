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
        self.buffer = Buffer(pv_list, 36000, logging_kwargs)

    def __call__(self):
        if self.logger is None:
            self.logger = create_worker_logger(**self.logging_kwargs)

        self.logger.debug(f"number of pvs running on: {len(self.pv_list)}")
        self.logger.debug("starting proecess_b data processing")

        while True:
            try:
                r = self.queue_one.get(timeout=0.05) # wait 50ms

                #self.logger.debug(f"ProcessB sees {r['iteration']}")
                #self.logger.debug(f"ProcessB sees {r}")
                #self.queue_two.put(r)

                if r is None: # enqueue a None to stop this process
                    break

                start = time.perf_counter()

                self.update_pv_values(r)
                result = self.find_candidates()

                # write to queue_2 for process_c.py to read

                # from process_c:
                # "r should have the structure (rf_input_tensor, bpm_input_tensor, rf_station)
                # where rf_input_tensor and bpm_input_tensor are tensors of size (1, 1066)
                # and (8, 1066) respectively, and rf_station is a string representing the PV name"
                fake_rf_input_tensor = np.random.rand(1, 1066).astype(np.float32)
                fake_bpm_input_tensor = np.random.rand(8, 1066).astype(np.float32)
                fake_pv_name = "fake_pv_name"

                fake_output_data = (fake_rf_input_tensor, fake_bpm_input_tensor, fake_pv_name)
                self.queue_two.put(fake_output_data)

                end = time.perf_counter()  # ⏱️ End profiling
                elapsed_ms = (end - start) * 1000
                self.logger.debug(f"process_b iteration took : {elapsed_ms:.2f} ms")

                if elapsed_ms > 1000: # have to be <= 1 sec
                    self.logger.warning(f"process_b iteration is slow!! : {elapsed_ms:.2f} ms")

            except Empty:
                continue

        self.logger.debug("ending data processing")

        #dir_name = f"buffer_txt_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        #self.logger.debug(f"dump dir: {dir_name}")
        #self.buffer.dump_to_human_readable(directory=dir_name)

        for handler in self.logger.handlers:
            handler.close()

    def update_pv_values(self, snapshot):

        iteration = snapshot.get("iteration", None)
        self.logger.debug(f"\niteration: {iteration}")

        for pv in self.pv_list:
            entries = snapshot.get(pv, [])
            num_entries = len(entries)
            #self.logger.debug(f"{pv}: {num_entries} entries in snapshot")

            if num_entries == 0:
                continue  # skip if no data

            times = np.empty(120, dtype=np.float64)
            values = np.empty(120, dtype=np.float64)

            for i, e in enumerate(entries):
                ts = e.get("timeStamp", {})
                times[i] = ts.get("secondsPastEpoch", 0)
                values[i] = e.get("value", np.nan)

        self.buffer.append(pv, values)

    def find_candidates(self):
        # add beam checks here
        return []

class Buffer:
    def __init__(self, pv_list: list[str], buffer_len: int = 36000, logging_kwargs: Optional[dict] = default_logging_kwargs):
        self.buffer_len = buffer_len
        self.pv_list = pv_list
        self.last_snapshot_time = 0 # for sanity check of snapshot validity
        self.starting_pv_time = 0 # store the earliest time in buffer, and assume all data is timed at 120hz
        self.buffer_map = {
            pv: np.empty(buffer_len, dtype=np.float64) for pv in self.pv_list
        }
        self.index = 0  # tracking next write index
        self.logging_kwargs = logging_kwargs
        self.logger = create_worker_logger(**self.logging_kwargs)
        self.logging_kwargs['logger_name'] = 'buffer'

    def dump_to_human_readable(self, directory: str = "buffer_dump_txt"):
        # individual human-readable .txt file per pv, each line has one float val
        os.makedirs(directory, exist_ok=True)

        for pv in self.pv_list:
            valid_data = self.buffer_map[pv][:self.index]
            pv = pv[5:] # get rid of "ca://"
            filepath = os.path.join(directory, f"{pv.replace(':', '_')}.txt")

            self.logger.debug(f"writing dump file {filepath} for {pv}")
            with open(filepath, "w") as f:
                for v in valid_data:
                    f.write(f"{v}\n")

    def append(self, key: str, values: np.ndarray):
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer")

        #if len(values) != 120:
        #    raise ValueError(f"Expected array of length 120, got {len(values)}")

        idx = self.index
        buf = self.buffer_map[key]

        #if idx == 0:
            #self.starting_pv_time = time

        if idx + 120 <= self.buffer_len:
            # have enough room without shifting, just write to next open index (this only happens during initial buffer fill-up)
            self.logger.debug(f"initial filling of buffer, curr index {idx}")
            buf[idx:idx+120] = values
            self.index += 120
        else:
            # shift left and append to the end
            self.logger.debug(f"buffer is full, removing oldest second of data")
            buf[:-120] = buf[120:]
            buf[-120:] = values
            self.index = self.buffer_len
            #self.starting_pv_time = time

    def get(self, key: str):
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer")
        return self.buffer_map[key][:self.index]

    def clear(self, key: str):
        if key not in self.buffer_map:
            raise KeyError(f"key '{key}' not found in buffer.")
        self.buffer_map[key][:] = np.empty(self.buffer_len, dtype=np.float64)
        self.index = 0