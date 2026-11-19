"""Reference (pre-optimisation) implementations kept to prove the fast code is exact."""

import numpy as np


def iou_matrix(a, b, valid):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    result = np.zeros((len(a), len(b)), np.float64)
    counts_a = np.array([(x & valid).sum() for x in a])
    counts_b = np.array([(x & valid).sum() for x in b])
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            intersection = (first & second & valid).sum()
            result[i, j] = intersection / max(1, counts_a[i] + counts_b[j] - intersection)
    return result
