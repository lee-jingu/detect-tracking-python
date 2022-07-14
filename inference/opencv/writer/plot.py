from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    color = np.asarray(color, dtype="uint8")
    return color


def draw_xyxy_box(img: Image, box: list[int], label: str, box_id: int):
    """
        img : input img
        box : xyxy box
        label : box label
        box_id : box id
    """
    x0, y0, x1, y1 = map(int, box)
    _color = get_color(box_id)
    color = _color.tolist()
    txt_bk_color = (_color * 0.7).astype(np.uint8).tolist()
    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    tl = 1
    tf = max(tl - 1, 1)

    # bounding box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, tl * 2)

    # label box
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    cbox1 = [x0, y0]
    cbox2 = [x0 + t_size[0], y0 - t_size[1] - 3]
    cv2.rectangle(img, cbox1, cbox2, txt_bk_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (cbox1[0], cbox1[1] - 2),
        0,
        tl / 3,
        txt_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

def draw_key_point(img: Image, point_id: int, point: tuple[int, int], size: int = 5):
    """
        img : input img
        point_id : point id
        point : xy point
    """
    x, y = map(int, point)
    _color = get_color(int(point_id))
    color = _color.tolist()
    cv2.circle(img, (x, y), size, color, -1)