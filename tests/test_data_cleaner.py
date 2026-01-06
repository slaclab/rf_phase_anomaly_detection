import numpy as np
import pytest
from data_bucketer import DataBucketer
from run_config import SAMPLES_PER_SECOND, NANOSECS_IN_1_SEC


@pytest.fixture
def data_bucketer():
    return DataBucketer(SAMPLES_PER_SECOND)


rng = np.random.default_rng(42)

## shared constants
STARTING_TIMESTAMP = 1_000_000_000
ONE_TIMESTEP = int(NANOSECS_IN_1_SEC / SAMPLES_PER_SECOND)
LAST_BUCKET_TIMESTAMP = STARTING_TIMESTAMP + (SAMPLES_PER_SECOND - 1) * ONE_TIMESTEP
BUCKET_TIMESTAMP_ARRAY = np.linspace(STARTING_TIMESTAMP, LAST_BUCKET_TIMESTAMP, SAMPLES_PER_SECOND, dtype=int)


## test case 1: two PVs with complete data
def generate_test_case_no_missing_data():
    def gen_pv_data():
        return (
            np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64),
            BUCKET_TIMESTAMP_ARRAY + rng.integers(-100, 100, size=SAMPLES_PER_SECOND),
        )

    data_map_with_per_pv_timestamps = {
        "PV1": gen_pv_data(),
        "PV2": gen_pv_data(),
    }

    expected_values = {
        "PV1": np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64),
        "PV2": np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64),
        "pv_timestamps_ns": BUCKET_TIMESTAMP_ARRAY,
    }

    return data_map_with_per_pv_timestamps, expected_values


## test case 2: one PV with full data, one PV missing some data (forward-filling needed)
def generate_test_case_forward_fill_missing_sections_of_snapshot():
    # pv1 has values for all expected timestamps
    pv1_values = np.arange(SAMPLES_PER_SECOND, dtype=np.float64)
    pv1_timestamps = BUCKET_TIMESTAMP_ARRAY + rng.integers(-100, 100, size=SAMPLES_PER_SECOND)

    # pv2 is missing some manually selected data-points
    pv2_values = np.arange(SAMPLES_PER_SECOND, dtype=np.float64)
    pv2_timestamps = BUCKET_TIMESTAMP_ARRAY + rng.integers(-100, 100, size=SAMPLES_PER_SECOND)

    # let's have 3 sections of missing data
    forward_fill_indices = [20, 21, 22, 53, 54, 80]

    pv2_values = np.delete(pv2_values, forward_fill_indices)
    pv2_timestamps = np.delete(pv2_timestamps, forward_fill_indices)

    data_map_with_per_pv_timestamps = {
        "PV1": (pv1_values, pv1_timestamps),
        "PV2": (pv2_values, pv2_timestamps),
    }

    # pv2's data is expected to forward-fill the missing values
    pv2_expected_values = pv1_values.copy()
    for idx in forward_fill_indices:
        pv2_expected_values[idx] = pv2_expected_values[idx - 1]

    expected_values = {
        "PV1": pv1_values,
        "PV2": pv2_expected_values,
        "pv_timestamps_ns": BUCKET_TIMESTAMP_ARRAY,
    }

    return data_map_with_per_pv_timestamps, expected_values


## test case 3: one PV with full data, one PV is completely empty (forward-filling from prev snapshot needed)
def generate_test_case_forward_fill_empty_pv_from_prev_snapshot():
    # pv1 has values for all expected timestamps
    pv1_values = np.arange(0, SAMPLES_PER_SECOND, dtype=np.float64)
    pv1_timestamps = BUCKET_TIMESTAMP_ARRAY + rng.integers(-100, 100, size=SAMPLES_PER_SECOND)

    # pv2 is completely empty
    pv2_values = np.array([], dtype=np.float64)
    pv2_timestamps = np.array([], dtype=int)

    data_map_with_per_pv_timestamps = {
        "PV1": (pv1_values, pv1_timestamps),
        "PV2": (pv2_values, pv2_timestamps),
    }

    expected_values = {
        "PV1": pv1_values,
        "PV2": np.full(
            SAMPLES_PER_SECOND, 42.0
        ),  # we expect to fully forward-fill the last value from the prev snapshot (42 is arbitrarily chosen)
        "pv_timestamps_ns": BUCKET_TIMESTAMP_ARRAY,
    }

    return data_map_with_per_pv_timestamps, expected_values


## test case runner
@pytest.mark.parametrize(
    "data_map_with_per_pv_timestamps, expected_values",
    [
        generate_test_case_no_missing_data(),
        generate_test_case_forward_fill_missing_sections_of_snapshot(),
        generate_test_case_forward_fill_empty_pv_from_prev_snapshot(),
    ],
)
def test_bucket_data_multiple_pvs(data_bucketer, data_map_with_per_pv_timestamps, expected_values):
    # represents the last values from the prev snapshot (to test forward-filling from prev snapshot)
    last_vals_prev_snapshot = {
        "PV1": 1.0,
        "PV2": 42.0,  # arbitrarily chosen
    }

    cleaned = data_bucketer.bucket_and_forward_fill_data(data_map_with_per_pv_timestamps, last_vals_prev_snapshot, STARTING_TIMESTAMP)

    for pv, expected in expected_values.items():
        assert pv in cleaned, f"Missing pv in cleaned data: {pv}"

        actual = cleaned[pv]
        assert isinstance(actual, np.ndarray), f"cleaned data for '{pv}' is not an arr (function return type is wrong)"
        assert actual.shape == expected.shape, (
            f"cleaned data for '{pv}' is not correct shape: expected {expected.shape} but got {actual.shape}"
        )

        # print array of diff values, makes things easier to debug when using 120 len arrays
        if not np.allclose(actual, expected):
            diff = actual - expected
            print(f"\ncleaned data for '{pv}' does not match the expected!")
            print(f"actual arr:\n{actual}")
            print(f"expected arr:\n{expected}")
            print(f"diff arr:\n{diff}")
            assert False, f"{pv} values mismatch"
