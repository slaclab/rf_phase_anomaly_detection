import numpy as np
from snapshot_fixer import SnapshotFixer
from beam_check_config import NANOSECS_IN_1_SEC
from snapshot_fixer import get_timestamp_ns


def test_fix_snapshot_basic():
    """
    Test handling when a snapshot contains data-points that we expect in a later snapshot. (based on timestamps)
    """

    pv_list = ["pv1", "pv2"]
    fixer = SnapshotFixer(pv_list)

    # simulate snapshot's pv-entry data
    def make_entry(ts_ns: int, value: float) -> dict:
        seconds = ts_ns // NANOSECS_IN_1_SEC  # // is int division
        nanoseconds = ts_ns % NANOSECS_IN_1_SEC
        return {
            "timeStamp": {
                "secondsPastEpoch": seconds,
                "nanoseconds": nanoseconds,
            },
            "value": value,
        }

    saved_for_later_val = 2.0
    saved_for_later_ts = 110
    raw_snapshot = {
        "iteration": 22,
        "timestamp": 5000,
        "pv1": [
            make_entry(10, 1.0),  # should be in curr snapshot
            make_entry(saved_for_later_ts, saved_for_later_val),  # should be saved for future snapshot
        ],
        "pv2": [
            make_entry(20, 3.0),  # should be in curr snapshot
        ],
    }

    fixed = fixer.fix_snapshot(raw_snapshot, 0, 100)

    # only current-window data is returned from `fix_snapshot()`
    assert "pv1" in fixed
    assert "pv2" in fixed

    np.testing.assert_array_equal(fixed["pv1"][0], np.array([1.0]))
    np.testing.assert_array_equal(fixed["pv2"][0], np.array([3.0]))

    # 2nd entry for pv1 should be left in temp_storage
    assert len(fixer.temp_storage["pv1"]) == 1
    assert len(fixer.temp_storage["pv2"]) == 0

    # make sure the stored entry is the one from the future
    future_entry = fixer.temp_storage["pv1"][0]
    assert future_entry["value"] == saved_for_later_val

    assert get_timestamp_ns(future_entry) == saved_for_later_ts
