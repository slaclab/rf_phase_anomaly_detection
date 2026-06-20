import os
import json

import numpy as np

from run_config import CANDIDATE_SAVE_DIRECTORY

from typing import Tuple


def do_nothing(*args, **kwargs):
    pass


def check_save_flag_file(func):
    sff = os.path.join(CANDIDATE_SAVE_DIRECTORY, "save_flag_file.txt")
    if os.path.exists(sff):
        with open(sff, 'r') as f:
            # the file should contain only True or False and nothing else
            save_flag = f.read().rstrip().lower() == 'true'
        if save_flag:
            return func
    return do_nothing  # do nothing without the file existing and flag set True


@check_save_flag_file
def save_snapshot(
        snapshot: dict,
        prefix: str,  # should be 'raw' or 'pruned'
        iteration: int,
):
    full_dir = os.path.join(CANDIDATE_SAVE_DIRECTORY, f"{prefix}_snapshots")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    fn = os.path.join(full_dir, f"{prefix}_snapshot_{iteration:04d}.json")
    with open(fn, 'w') as f:
        json.dump(snapshot, f)


@check_save_flag_file
def save_fixed_snapshot(
        fixed_data: dict[str, Tuple[np.ndarray, np.ndarray]],
        latest_data: dict[str, float],
        iteration: int,
):
    full_dir = os.path.join(CANDIDATE_SAVE_DIRECTORY, f"fixed_snapshots")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    fn = os.path.join(full_dir, f"fixed_snapshot_{iteration:04d}.npz")
    np.savez(fn, data=fixed_data, latest=latest_data)


@check_save_flag_file
def save_bucketed_snapshot(
        bucketed_data: dict[str, np.ndarray],
        iteration: int
):
    full_dir = os.path.join(CANDIDATE_SAVE_DIRECTORY, f"bucketed_snapshots")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    fn = os.path.join(full_dir, f"bucketed_snapshot_{iteration:04d}.npz")
    np.savez(fn, **bucketed_data)


@check_save_flag_file
def save_sliding_windows(
        windows: dict[str, np.ndarray],
        iteration: int
):
    full_dir = os.path.join(CANDIDATE_SAVE_DIRECTORY, f"sliding_windows")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    fn = os.path.join(full_dir, f"sliding_windows_{iteration:04d}.npz")
    np.savez(fn, **windows)
