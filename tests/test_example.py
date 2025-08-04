import pytest
import numpy as np

from k2eg_spoofer import K2EGSpoofer
from buffer import Buffer
from beam_check_config import BUFFER_LENGTH
from mp_logging import default_logging_kwargs
from k2eg_process import read_pv_list_from_file
from sliding_window import SlidingWindowArray


@pytest.fixture
def pv_list():
    return read_pv_list_from_file("resources/pv_list.txt")


@pytest.fixture
def spoofed_data(pv_list):
    spoofer = K2EGSpoofer(
        pv_configs=[{"name": name, "rate_hz": 120, "drop_rate": 0.0} for name in pv_list], n_emits=1, emit_rate_hz=1
    )
    return list(spoofer())


@pytest.fixture
def buffer(pv_list, spoofed_data):
    buffer = Buffer(pv_list, BUFFER_LENGTH, default_logging_kwargs)
    for emission in spoofed_data:
        buffer.update(emission)
    return buffer


def test_example_using_buffer(buffer, pv_list):
    print(buffer.get(pv_list[0]))
    assert len(buffer.get(pv_list[1])) == 120
    """
    for pv in pv_list:
        print(pv)
        print(len(buffer.get(pv)))
    """


def test_example_modifying_buffer_data(buffer, pv_list):
    test_data = SlidingWindowArray(BUFFER_LENGTH, dtype=np.float64)
    test_data.put(np.random.choice([False, True], size=120))
    buffer.data_map["beam_checks"] = test_data

    print(buffer.get("beam_checks"))
