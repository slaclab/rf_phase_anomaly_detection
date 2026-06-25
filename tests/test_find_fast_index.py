import pytest
import numpy as np

from anomaly_candidate import find_fast_index
from run_config import CANDIDATE_LOOKBACK_WINDOW_LENGTH


@pytest.fixture
def make_easy_score_window():
    return (np.arange(CANDIDATE_LOOKBACK_WINDOW_LENGTH) > 10).astype(float) * 5.0


@pytest.fixture
def make_short_score_window():
    return (np.arange(9) > 10).astype(float) * 5.0


@pytest.fixture
def make_flat_score_window():
    return np.zeros(20)


def test_find_fast_index(make_easy_score_window):
    found_index = find_fast_index(make_easy_score_window)
    assert found_index == 11


def test_find_short_index(make_short_score_window):
    score_window = make_short_score_window
    found_index = find_fast_index(score_window)
    assert len(score_window) == found_index


def test_find_flat_index(make_flat_score_window):
    score_window = make_flat_score_window
    found_index = find_fast_index(score_window)
    assert found_index == 10
