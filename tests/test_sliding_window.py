import numpy as np
import pytest

from sliding_window import SlidingWindowArray


@pytest.fixture
def buffer():
    return SlidingWindowArray(buffer_len=5, dtype=np.float64, pv_name="test_pv")


def test_initial_state(buffer):
    assert buffer.index == 0
    assert buffer.buffer_len == 5
    assert buffer.pv_name == "test_pv"
    assert isinstance(buffer.data, np.ndarray)
    assert buffer.data.shape == (5,) # np shapes are tuples, (5,) is a tupe, (5) is not and fails this comparison
    assert buffer.data.dtype == np.float64
    assert len(buffer.data) == 5


def test_put_and_get(buffer):
    buffer.put(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(buffer.get(), [1.0, 2.0])
    assert buffer.index == 2


def test_get_with_bounds(buffer):
    buffer.put(np.array([10, 20, 30]))
    np.testing.assert_array_equal(buffer.get(0, 2), [10, 20])
    np.testing.assert_array_equal(buffer.get(-1), [30])
    np.testing.assert_array_equal(buffer.get(1, None), [20, 30])


def test_put_too_many_expect_error(buffer):
    with pytest.raises(ValueError):
        buffer.put(np.array([1, 2, 3, 4, 5, 6]))


def test_buffer_shift_when_full(buffer):
    buffer.put(np.array([1, 2, 3]))
    buffer.put(np.array([4, 5]))
    assert buffer.is_full()

    buffer.put(np.array([6]))
    np.testing.assert_array_equal(buffer.get(), [2, 3, 4, 5, 6])

    buffer.put(np.array([7, 8, 9]))
    np.testing.assert_array_equal(buffer.get(), [5, 6, 7, 8, 9])

    buffer.put(np.array([10, 11, 12, 13, 14]))
    np.testing.assert_array_equal(buffer.get(), [10, 11, 12, 13, 14])


def test_get_invalid_indices_expect_error(buffer):
    buffer.put(np.array([1, 2, 3]))
    with pytest.raises(IndexError):
        buffer.get(-1, 5)
    with pytest.raises(IndexError):
        buffer.get(2, 1)
    with pytest.raises(IndexError):
        buffer.get(0, 10)


def test_clear(buffer):
    buffer.put(np.array([1, 2, 3]))
    buffer.clear()
    assert buffer.index == 0
    assert np.all(np.isnan(buffer.data))


def test_len_operator(buffer):
    assert len(buffer) == 0
    buffer.put(np.array([1, 2]))
    assert len(buffer) == 2