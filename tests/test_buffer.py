import pytest
import numpy as np

from buffer import Buffer
from beam_check_config import SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC
from k2eg_process import read_pv_list_from_file


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


def test_buffer_update_basic(pv_list):
    """Test that Buffer.update correctly processes two 1-second snapshots."""
    buffer_len = 36000
    buffer = Buffer(pv_list, buffer_len, SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC)

    # Generate 120Hz timestamps for 2 seconds
    ts_offset = NANOSECS_IN_1_SEC // SAMPLES_PER_SECOND
    timestamps_1 = [i * ts_offset for i in range(SAMPLES_PER_SECOND)]
    timestamps_2 = [(i * ts_offset) + NANOSECS_IN_1_SEC for i in range(SAMPLES_PER_SECOND)]

    snapshot_1 = {pv: [make_entry(ts, float(i)) for i, ts in enumerate(timestamps_1)] for pv in pv_list} | {'iteration': 0, 'timestamp': 0}

    snapshot_2 = {pv: [make_entry(ts, float(i + 1000)) for i, ts in enumerate(timestamps_2)] for pv in pv_list} | {'iteration': 1000, 'timestamp': 1000}

    # first update
    index_change, length_of_update = buffer.update(snapshot_1)
    assert length_of_update == SAMPLES_PER_SECOND
    assert index_change == 0
    assert buffer.index == SAMPLES_PER_SECOND

    # second update
    index_change, length_of_update = buffer.update(snapshot_2)
    assert length_of_update == SAMPLES_PER_SECOND
    assert index_change == 0
    assert buffer.index == 2 * SAMPLES_PER_SECOND

    # expect first 120 values to be 0.0 to 119.0, second 120 to be 1000.0 to 1119.0
    expected = np.concatenate(
        [
            np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64),
            np.arange(1000, 1000 + SAMPLES_PER_SECOND, dtype=np.float64),
        ]
    )

    for pv in pv_list:
        data = buffer.data_map[pv].get()[: buffer.index]
        assert isinstance(data, np.ndarray)
        assert len(data) == 2 * SAMPLES_PER_SECOND
        np.testing.assert_array_almost_equal(data, expected, decimal=5)

    # timestamps check
    ts_data = buffer.data_map["pv_timestamps_ns"].get()[: buffer.index]
    assert len(ts_data) == 2 * SAMPLES_PER_SECOND
    assert ts_data[0] == 0
    assert ts_data[-1] < 2 * NANOSECS_IN_1_SEC

    # Beam and score arrays should also match length
    assert len(buffer.data_map["beam_checks"]) == 2 * SAMPLES_PER_SECOND
    assert len(buffer.data_map["bpm_score_1"]) == 2 * SAMPLES_PER_SECOND
    assert len(buffer.data_map["bpm_score_20"]) == 2 * SAMPLES_PER_SECOND
