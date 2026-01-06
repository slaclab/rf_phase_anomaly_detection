import pytest
import logging
import numpy as np
from collections import deque

from buffer import Buffer, get_latest_time_point, get_value
from snapshot_fixer import SnapshotFixer, process_pv_from_snapshot
from run_config import SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from utilities import read_pv_list_from_file


@pytest.fixture
def pv_list():
    return read_pv_list_from_file("resources/pv_list.txt")


def make_entry(ts_ns: int, value: float) -> dict:
    return {
        "timeStamp": {
            "secondsPastEpoch": ts_ns // NANOSECS_IN_1_SEC,
            "nanoseconds": ts_ns % NANOSECS_IN_1_SEC,
        },
        "value": value,
    }


def test_data_fixer(pv_list):
    """Test that SnapshotFixer can correctly fix a single PV's worth of data"""
    fixer = SnapshotFixer(pv_list=pv_list)

    ts_offset = NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
    timestamps = [i * ts_offset for i in range(SAMPLES_PER_SECOND)]
    values = np.arange(0, 0 + len(timestamps), dtype=np.float64)

    snapshot = {pv: [make_entry(ts, v) for ts, v in zip(timestamps, values)] for pv in pv_list} | {
        'iteration': 0, 'timestamp': 0}

    start_time = 0
    end_time = start_time + NANOSECS_IN_1_SEC
    pv_name = pv_list[0]
    fixed_snapshot_data, _ = process_pv_from_snapshot(
        pv_name=pv_name,
        snapshot_of_pv=snapshot[pv_name],
        temp_storage=deque(),
        start_time=start_time,
        end_time=end_time,
        logger=logging.getLogger('SnapshotFixer_test')
    )
    assert all(fixed_snapshot_data[0] == values)
    assert all(fixed_snapshot_data[1] == timestamps)

    bucket_timestamps = fixer.data_bucketer._generate_bucket_timestamps(start_time, NANOSECS_IN_1_SEC)
    bucket_values = fixer.data_bucketer._map_values_to_buckets(
        fixed_snapshot_data[0],
        fixed_snapshot_data[1],
        bucket_timestamps
    )
    assert all(bucket_values == values)


def test_buffer_update_basic(pv_list):
    """Test that Buffer.update correctly processes the first three 1-second snapshots."""
    buffer_len = 36000
    buffer = Buffer(pv_list, buffer_len, SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC)

    # Generate 120Hz timestamps for 3 seconds
    ts_offset = NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
    timestamps_1 = [i * ts_offset for i in range(SAMPLES_PER_SECOND)]
    timestamps_2 = [(i * ts_offset) + NANOSECS_IN_1_SEC for i in range(SAMPLES_PER_SECOND)]
    timestamps_3 = [(i * ts_offset) + 2 * NANOSECS_IN_1_SEC for i in range(SAMPLES_PER_SECOND)]

    snapshot_1 = {pv: [make_entry(ts, float(i)) for i, ts in enumerate(timestamps_1)] for pv in pv_list} | {'iteration': 0, 'timestamp': 0}

    snapshot_2 = {pv: [make_entry(ts, float(i + 1000)) for i, ts in enumerate(timestamps_2)] for pv in pv_list} | {'iteration': 1000, 'timestamp': 1000}

    snapshot_3 = {pv: [make_entry(ts, float(i + 2000)) for i, ts in enumerate(timestamps_3)] for pv in pv_list} | {'iteration': 2000, 'timestamp': 2000}

    # first update - does not change buffer
    index_change, length_of_update = buffer.update(snapshot_1)
    assert length_of_update == 0
    assert index_change == 0
    assert buffer.index == 0

    # second update
    index_change, length_of_update = buffer.update(snapshot_2)
    assert length_of_update == SAMPLES_PER_SECOND
    assert index_change == 0
    assert buffer.index == SAMPLES_PER_SECOND

    # third update
    index_change, length_of_update = buffer.update(snapshot_3)
    assert length_of_update == SAMPLES_PER_SECOND
    assert index_change == 0
    assert buffer.index == 2 * SAMPLES_PER_SECOND

    # expect first 120 values to be 1000 to 1119, second 120 to be 2000.0 to 2119.0
    expected = np.concatenate(
        [
            # np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64),
            np.arange(1000, 1000 + SAMPLES_PER_SECOND, dtype=np.float64),
            np.arange(2000, 2000 + SAMPLES_PER_SECOND, dtype=np.float64),
        ]
    )

    for pv in pv_list:
        data = buffer.data_map[pv].get()[:buffer.index]
        import json
        with open('test_data.json', 'w') as jf:
            json.dump(list(data), jf)
        assert isinstance(data, np.ndarray)
        assert len(data) == 2 * SAMPLES_PER_SECOND
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

    # timestamps check
    ts_data = buffer.data_map["pv_timestamps_ns"].get()[:buffer.index]
    assert len(ts_data) == 2 * SAMPLES_PER_SECOND
    assert np.isclose(ts_data[0], NANOSECS_IN_1_SEC)
    assert ts_data[-1] < 3 * NANOSECS_IN_1_SEC

    # Beam and score arrays should also match length
    assert len(buffer.data_map["beam_checks"]) == 2 * SAMPLES_PER_SECOND
    assert len(buffer.data_map["bpm_score_1"]) == 2 * SAMPLES_PER_SECOND
    assert len(buffer.data_map["bpm_score_20"]) == 2 * SAMPLES_PER_SECOND


def test_buffer_update_slow_data(pv_list):
    """Tests a buffer that receives data at 5 Hz and arbitrarily"""
    pv_list = ['a', 'b']

    values_a = np.arange(0, 5, dtype=np.float64)
    timestamps_a = 1e9 * values_a / 5
    timestamps_b = [int(0.45e9)]
    values_b = [10]
    snapshot_1 = {
                     'a': [make_entry(x, float(y)) for x, y in zip(timestamps_a, values_a)],
                     'b': [make_entry(x, y) for x, y in zip(timestamps_b, values_b)],
                 } | {'iteration': 0, 'timestamp': 0}

    values_a = np.arange(6, 10, dtype=np.float64)
    timestamps_a = 1e9 * values_a / 5 + 0.2
    timestamps_b = [int(1.65e9), int(2.10e9)]
    values_b = [11, 12]
    snapshot_2 = {
                     'a': [make_entry(x, float(y)) for x, y in zip(timestamps_a, values_a)],
                     'b': [make_entry(x, float(y)) for x, y in zip(timestamps_b, values_b)],
                 } | {'iteration': 0, 'timestamp': 0}

    expected_buffer_a = np.array([
        4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
        4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
        4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 6., 6., 6., 6.,
        6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        6., 6., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.,
        7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 8., 8., 8., 8., 8., 8., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
        9.
    ])
    expected_buffer_b = np.array([
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 11., 11., 11.,
        11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
        11., 11., 11.
    ])

    buffer = Buffer(
        pv_list=pv_list,
        buffer_len=36000,
        snapshot_length=120,
        snapshot_period_ns=int(1e9),
    )

    self = buffer
    for snapshot in [snapshot_1, snapshot_2]:
        # if this is the first snapshot seen, get the oldest time across all pv data-points
        if self.time_of_first_data == -1:
            min_ts = get_latest_time_point(snapshot, self.pv_list)
            # the first data is expected to show up one bucket after the last sample found:
            self.time_of_first_data = min_ts + NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
        else:  # otherwise, process the data
            # get the expected time window (start and end) for this snapshot
            start_time = self.time_of_first_data + self.num_snapshots_processed * self.snapshot_period_ns
            end_time = start_time + self.snapshot_period_ns

            # makes sure that all new data is within the above time window, also gets the latest datapoint
            # that should have already been sent, if any
            fixed_snapshot_data, new_latest_values = self.fixer.fix_snapshot(snapshot, start_time, end_time)
            # copy any new latest values into memory
            for key, value in new_latest_values.items():
                self.prev_snapshot_val_map[key] = value
            # bucket the data into 120 Hz buckets
            fixed_and_bucketed_snapshot_data = self.fixer.bucket_snapshot_data(
                fixed_snapshot_data, self.prev_snapshot_val_map, start_time, end_time
            )

            for pv, values in fixed_and_bucketed_snapshot_data.items():
                self.data_map[pv].put(values)
            buffer.index += 120

        for pv in self.pv_list:
            prev_snapshot_val = get_value(snapshot[pv][-1], self.logger)
            self.prev_snapshot_val_map[pv] = prev_snapshot_val

    assert all(buffer.get('a') == expected_buffer_a)
    assert all(buffer.get('b') == expected_buffer_b)
    assert len(buffer.fixer.temp_storage['a']) == 0
    assert len(buffer.fixer.temp_storage['b']) == 1
