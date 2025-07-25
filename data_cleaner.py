from typing import Optional, Tuple
import numpy as np

from mp_logging import create_worker_logger, default_logging_kwargs


class DataCleaner:
    """
    Cleans and aligns per-PV timestamped data into fixed time buckets.

    This class can take a snapshot of PV data, where each PV might have irregular or missing timestamps,
    and convert it into an array of values aligned to evenly spaced 1/`samples_per_second` bucket intervals,

    Missing values are forward-filled using the previous value from the same PV.
    If the missing value is the first entry in a bucket, it uses the last known value from the prev snapshot.

    Usage
    -----
    data_cleaner = DataCleaner(SAMPLES_PER_SECOND)
    ...
    data_map_bucketed = data_cleaner.clean_data(data_map, prev_snapshot_map, start_time)
    """

    def __init__(self, samples_per_second: int, logging_kwargs: Optional[dict] = default_logging_kwargs):
        """
        Parameters
        ----------
        samples_per_second : int
            The number of data samples expected per second.
        """
        logging_kwargs["logger_name"] = "data_cleaner"
        self.logger = create_worker_logger(**logging_kwargs)

        self.samples_per_second = samples_per_second

        self.logger.info(f"Initialized DataCleaner with samples_per_second={samples_per_second}")

    def clean_data(
        self,
        data_map_with_per_pv_timestamps: dict[str, Tuple[np.ndarray, np.ndarray]],
        prev_snapshot_val_map: dict[str, np.float64],  # map of pv to last value in prev snapshot (for forward-filling)
        bucket_arr_start_time: int,
    ) -> dict[str, np.ndarray]:
        """
        Cleans and buckets timestamped pv data into evenly spaced time-buckets of `1/self.amples_per_second`

        Parameters
        ----------
        data_map_with_per_pv_timestamps : dict
            A mapping from PV name to (values, timestamps) arrays.
        prev_snapshot_val_map : dict
            A mapping from PV name to the most recent value from the previous snapshot, for forward-filling.
        bucket_arr_start_time : int
            The nanosecond timestamp indicaitng the start of this bucket interval.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping each PV to its cleaned array of values, and includes a shared "pv_timestamps_ns" array.
        """
        self.logger.debug("Starting to clean data...")
        self.logger.debug(f"Buckets array starting time: {bucket_arr_start_time}")

        result_map = {}

        # generate evenly spaced target timestamps to use as bucket centers
        duration_ns = int(1e9)  # 1 second in nanoseconds
        bucket_timestamps = self._generate_bucket_timestamps(bucket_arr_start_time, duration_ns)
        result_map["pv_timestamps_ns"] = bucket_timestamps
        self.logger.debug(f"Generated {len(bucket_timestamps)} bucket timestamps")

        for pv, (values, timestamps_ns) in data_map_with_per_pv_timestamps.items():
            # self.logger.debug(f"Processing PV: {pv} with {len(values)} values")
            # assign each value to the closest bucket timestamp
            bucket_values = self._map_values_to_buckets(values, timestamps_ns, bucket_timestamps)
            # self.logger.debug(f"Mapped PV '{pv}' values to buckets with {np.count_nonzero(np.isnan(bucket_values))} non-NaN entries")
            # fill in any missing values using prior vals or previous snapshot (if no prev val in curr timestamp)
            bucket_values = self._forward_fill(bucket_values, pv, prev_snapshot_val_map)
            # self.logger.debug(f"Forward-filled PV '{pv}' resulting in {np.count_nonzero(~np.isnan(bucket_values))} NaN entries")
            result_map[pv] = bucket_values

        self.logger.debug("Done with cleaning data")
        return result_map

    def _generate_bucket_timestamps(self, start_ts: int, duration_ns: int) -> np.ndarray:
        """
        Generate evenly spaced timestamps over a given duration.

        Parameters
        ----------
        start_ts : int
            Starting timestamp in nanoseconds.
        duration_ns : int
            Duration of the bucket range in nanoseconds.

        Returns
        -------
        np.ndarray
            An array of bucket timestamps spaced evenly across the duration.
        """
        step = duration_ns / self.samples_per_second
        return np.array([int(start_ts + (i * step)) for i in range(self.samples_per_second)], dtype=np.int64)

    def _map_values_to_buckets(
        self,
        values: np.ndarray,
        timestamps_ns: np.ndarray,
        bucket_timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Map timestamped values into the nearest bucket timestamps.

        Parameters
        ----------
        values : np.ndarray
            The array of data values.
        timestamps_ns : np.ndarray
            The corresponding timestamps for the values.
        bucket_timestamps : np.ndarray
            The array of target timestamps for bucketing.

        Returns
        -------
        np.ndarray
            An array of length `samples_per_second` with values assigned to nearest time bucket.
        """
        bucket_values = np.full(self.samples_per_second, np.nan, dtype=np.float64)

        for val, ts in zip(values, timestamps_ns):
            # to find the bucket closest to our real timestamp,
            # we can subtract our curr time from all the bucket times to find which bucket is closest (smallest diff)
            idx = np.argmin(np.abs(bucket_timestamps - ts))

            if 0 <= idx < self.samples_per_second:
                bucket_values[idx] = val  # last write overrides if multiple values map to same bucket
        return bucket_values

    def _forward_fill(
        self, bucket_values: np.ndarray, pv: str, prev_snapshot_val_map: dict[str, np.float64]
    ) -> np.ndarray:
        for i in range(self.samples_per_second):
            if np.isnan(bucket_values[i]):
                if i == 0:
                    # use previous snapshot value for first position
                    bucket_values[i] = prev_snapshot_val_map[pv]
                else:
                    # use previous value in this array for all other positions
                    bucket_values[i] = bucket_values[i - 1]
        return bucket_values
