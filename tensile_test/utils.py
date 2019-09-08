"""`tensile_test.utils.py`"""

from pathlib import Path

import numpy as np


def find_nearest_index(arr, val):
    'Find the 1D array index whose value is closest to some value.'
    return np.nanargmin(np.abs(arr - val))


def read_non_uniform_csv(path, delimiter=',', skip_rows=0, header_row=1):
    'Load CSV file with variable length columns into a numpy array.'

    path = Path(path)

    arrs = []
    headers = None
    with path.open() as handle:
        for ln_idx, ln in enumerate(handle):

            ln = ln.strip()

            if header_row is not None and ln_idx == header_row:
                headers = [i for i in ln.split(delimiter)]

            if ln_idx < skip_rows:
                continue

            ln_arr = []
            for i in ln.split(delimiter):
                try:
                    i_parse = float(i)
                except ValueError:
                    i_parse = np.nan
                ln_arr.append(i_parse)

            arrs.append(ln_arr)

    arrs = np.array(arrs)

    if headers:
        return headers, arrs
    else:
        return arrs


def nan_array_to_list(arr):
    return np.where(np.isnan(arr), None, arr).tolist()
