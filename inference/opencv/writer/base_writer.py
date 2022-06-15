from __future__ import annotations
import os

import cv2
import numpy as np

from inference.interface.reader import ReaderInterface
from inference.interface.writer import WriterInterface

class Writer(WriterInterface):
    """
    Writer class for writing data to a destination.
    """

    def __init__(self, name: str, width: int, height: int, fps: float = None):
        pass