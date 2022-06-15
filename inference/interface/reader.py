from __future__ import annotations
from abc import ABCMeta, abstractmethod

import numpy as np

class ReaderInterface(metaclass=ABCMeta):
    """
    Interface for reading data from a source.
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
    def fps(self) -> float:
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
    
    @property
    @abstractmethod
    def info(self) -> dict:
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
    def read(self) -> np.ndarray | None:
        """Returns next frame from the video if available
        Returns:
            Union[np.ndarry, None]: next frame if available, None otherwise.
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
    def __next__(self) -> np.ndarray:
        """Returns next frame from the video
        Raises:
            StopIteration: No more frames to read
        Returns:
            np.ndarray: frame read from video
        """
        ...

    @abstractmethod
    def __iter__(self) -> "ReaderInterface":
        """Returns iterable object for reading frames
        Returns:
            Iterable[IReader]: iterable object for reading frames
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Video's Info
        Returns:
            str: info
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Video's Info
        Returns:
            str: Info
        """
        ...

    @abstractmethod
    def __enter__(self) -> "ReaderInterface":
        """Returns Conext for "with" block usage
        Returns:
            ReaderInterface: Video Reader object
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