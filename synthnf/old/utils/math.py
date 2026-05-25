import numpy as np


def normalize(arr, nan_to_zero=False):
    min_arr = np.min(arr)
    max_arr = np.max(arr)
    if min_arr == max_arr and 0 <= min_arr <= 1:
        return arr
    res = (arr - min_arr) / (max_arr - min_arr)
    if nan_to_zero:
        res = np.nan_to_num(res)
    return res
