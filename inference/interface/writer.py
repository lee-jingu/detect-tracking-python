from __future__ import annotations
from abc import ABCMeta, abstractmethod

import numpy as np
from .reader import ReaderInterface

class WriterInterface(metaclass=ABCMeta):
    """
    Interface for writing data to a destination.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of Source
        Returns:
            str: name of source
        """
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Width of Source
        Returns:
            int: width of frame
        """
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Height of Source
        Returns:
            int: height of frame
        """
        ...

    @property
    @abstractmethod
    def fps(self) -> float | None:
        """FPS of Video
        Returns:
            float: fps of video
        """
        ...
    
    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total frames read
        Returns:
            int: read frames' count
        """
        ...

    @property
    @abstractmethod
    def seconds(self) -> float:
        """Total seconds read
        Returns:
            float: read frames' in seconds
        """
        ...

    @property
    @abstractmethod
    def minutes(self) -> float:
        """Total minutes read
        Returns:
            float: read frames' in minutes
        """
        ...
    

    @abstractmethod
    def is_open(self) -> bool:
        """Checks if video is still open and last read frame was valid
        Returns:
            bool: True if video is open and last frame was not None, false otherwise.
        """
        ...

    @abstractmethod
    def write_vid(self, frame: np.ndarray) -> None:
        """Write frame to output video
        Args:
            frame (np.ndarray): frame to write
        """
        ...
    
    @abstractmethod
    def write_img(self, frame: np.ndarray) -> None:
        """Write frame to output image
        Args:
            frame (np.ndarray): frame to write
        """
        ...

    @abstractmethod
    def draw_bbox(self, image: Image, output: tuple[list[int], str]) -> None:
        """Draw bounding box on image
        Args:
            image (Image): image to draw on
            output (tuple[list[int], str]): bounding box and label
        """
        ...
    
    def draw_key_points(self, image: Image, key_point: dict[str, str]) -> None:
        """Draw key points on image
        Args:
            image (Image): image to draw on
            key_point (dict[str, str]): key points and labels
        """
        ...

    def save_txt(self, output: list[str]) -> None:
        """Save output to text file
        Args:
            output (list[str]): output to save
        """
        ...
    
    def save_json(self, output: list[dict]) -> None:
        """Save output to json file
        Args:
            output (list[dict]): output to save
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release Resources
        """
        ...

    @abstractmethod
    def __del__(self) -> None:
        """Release Resources
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Writer's Info
        Returns:
            str: info
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Writer's Info
        Returns:
            str: info
        """
        ...

    @abstractmethod
    def __enter__(self) -> "WriterInterface":
        """Returns Conext for "with" block usage
        Returns:
            WriterInterface: Video Writer object
        """
        ...

    @abstractmethod
    def __exit__(self, exc_type: None, exc_value: None,
                 traceback: None) -> None:
        """Release resources before exiting the "with" block
        Args:
            exc_type (NoneType): Exception type if any
            exc_value (NoneType): Exception value if any
            traceback (NoneType): Traceback of Exception
        """
        ...
