from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Box:
    x0: int
    y0: int
    w: int
    h: int
    padding: tuple[int, int] = (0, 0)

    def __post_init__(self):
        self.xywh = (self.x0, self.y0, self.w, self.h)
        self.xyxy = (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)

@dataclass
class Person:
    id: int
    visible: Box
    full: Box
    head: Box
    key_point: list[tuple[int, int]]