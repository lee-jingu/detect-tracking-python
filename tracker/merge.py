from __future__ import annotations
from typing import Union

import numpy as np

def align(branch:np.ndarray, base:np.ndarray, shape:Union[list, tuple, np.ndarray]) -> np.ndarray:
    """
        branch : align base에 align을 시도하는 새로운 detection
        base ; 새로운 detection을 align 할 기존 tracks
        shape : distance normalizing를 위한 image width, height
    """
    if len(branch) == 0 or len(base) == 0:
        return [], [], []

    height, width = shape

    dwc = (
        np.absolute(
            ((base[:, None, 0] + base[:, None, 2]) - (branch[:, 0] + branch[:, 2]))
        )
        / 2
    )
    dh = np.absolute(base[:, None, 1] - branch[:, 1])
    dist = 2 - (dwc / width + dh / height)

    iw = np.maximum(
        0,
        np.minimum(base[:, None, 2], branch[:, 2])
        - np.maximum(base[:, None, 0], branch[:, 0]),
    )
    ih = np.maximum(
        0,
        np.minimum(base[:, None, 3], branch[:, 3])
        - np.maximum(base[:, None, 1], branch[:, 1]),
    )
    area = (branch[:, 2] - branch[:, 0]) * (branch[:, 3] - branch[:, 1] + 0.0)
    overlap = iw * ih / area

    overlapped_idx = [
        np.any((iw[i] > 0) & (ih[i] > 0) & (overlap[i] > 0.7)) for i in range(len(base))
    ]

    cost = overlap + dist
    aligned_ids = np.argmax(cost[overlapped_idx], 1)
    target_tid = base[overlapped_idx, 4]
    bboxes = branch[aligned_ids]
    tids = target_tid[: len(aligned_ids)]
    output = np.c_[bboxes, tids]

    return output