from __future__ import annotations
import os
import time

import cv2
import numpy as np

from inference.interface.reader import ReaderInterface
from inference.utils import get_sorted_alpanumeric_files

image_extensions = set(['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'])
class ImageReader(ReaderInterface):
    """
    Video Reading wrapper around Opencv-Backend
    """
    def __init__(self,
                 path: str,
                 batch_size: int | None = None,
                 dynamic_batch: bool = False,
                 width: int | None = None,
                 height: int | None = None,
                 fps: int = 30,
                 **kwargs) -> None:
        """Initiate Reader object
        Args:
            path (str): path to image dir or image file
            batch_size (int | None): number of frames to return (as one batch) for one read.
            Defaults to None will return images individually without batch axis.
            dynamic_batch (bool): if set to True then last batch of frames may have
            less than batch_size frames (depending on how many frames were left for last batch).
            If set to False, last batch may have some frames made up of zeros to match batch_size.
            Defaults to False.
        """
        # initiate props
        self._init_props()

        # set batch
        self._batch_size = batch_size
        self._dynamic_batch = dynamic_batch

        # set image size
        self._width = width
        self._height = height

        # update info with current video stream
        self._post_init(path)

    def _init_props(self) -> None:
        """Init all class properties to default values
        """
        self._name = None
        self._width = None
        self._height = None
        self._is_dir = False
        self._is_open = None
        self._info = None
        self._fps = 30
        self._frame_count = 0
        self._seconds = 0
        self._minutes = 0
        self._num_files = 0
        self._img_files = []
        self._batch_size = None
        self._dynamic_batch = False
        self._prev_process_time = time.time()

    def _post_init(self, path: str) -> None:
        """Update info property
        """
        if path.endswith('/'):
            path = path[:-1]
        self._name = os.path.basename(path)
        if os.path.isdir(path):
            self._is_dir = True
        elif self._name.endswith in image_extensions:
            self._name = self._name[:-4]
        
        # get image files
        self._img_files = get_sorted_alpanumeric_files(path, image_extensions)
        self._num_files = len(self._img_files)
        self._inference_time = 1 / self._fps * 1000
        
        # update info
        self._info = {
            "name": self._name,
            "width": self._width,
            "height": self._height,
            "frame": self._frame_count,
            "num_files": self._num_files,
        }

    @property
    def name(self) -> str:
        """Name of Video Source
        Returns:
            str: name of video source
        """
        return self._name

    @property
    def width(self) -> int:
        """Width of Video
        Returns:
            int: width of video frame
        """
        return self._width

    @property
    def height(self) -> int:
        """Height of Video
        Returns:
            int: height of video frame
        """
        return self._height

    @property
    def fps(self) -> float:
        """FPS of Image Source
        Returns:
            float: fps of image
        """
        return self._fps

    @property
    def info(self) -> dict:
        """Video information
        Returns:
            dict: info of width, height, fps and backend.
        """
        return self._info

    @property
    def frame_count(self) -> int:
        """Total frames read
        Returns:
            int: read frames' count
        """
        return self._frame_count

    @property
    def seconds(self) -> float:
        """Total seconds read
        Returns:
            float: read frames' in seconds
        """
        return (self._frame_count / self._fps) if self._fps else 0

    @property
    def minutes(self) -> float:
        """Total minutes read
        Returns:
            float: read frames' in minutes
        """
        return self.seconds / 60.0

    def is_open(self) -> bool:
        """Checks if image show is still open and last read frame was valid
        Returns:
            bool: True if video is open and last frame was not None, false otherwise.
        """
        return self._is_open

    def read_frame(self) -> np.ndarray | None:
        """Returns next frame from the video if available
        Returns:
            Union[np.ndarry, None]: next frame if available, None otherwise.
        """

        frame = cv2.imread(self._img_files[self._frame_count])
        self._frame_count += 0 if frame is None else 1
        self._is_open = frame is not None
        return frame

    def read_batch(self) -> np.ndarray | None:
        """Returns next batch of frames from the video if available
        Returns:
            np.ndarry | None: next batch if available, None otherwise.
        """
        if not self.is_open():
            return None

        # pre-allocate batch
        batch = np.zeros((self._batch_size, self.height, self.width, 3), dtype="uint8")

        # fill batch
        for i in range(self._batch_size):
            # read frame
            frame = self.read_frame()

            # stop process, no frames left
            if frame is None:
                # decrm index because this frame was empty
                i -= 1
                break

            # add to batch
            batch[i] = frame

        return batch[:i + 1] if self._dynamic_batch else batch

    def read(self) -> np.ndarray | None:
        """Returns next frame or batch of frames from the video if available
        Returns:
            np.ndarry | None: next frame or batch of frames if available, None otherwise.
        """
        if self._batch_size is None:
            return self.read_frame()
        return self.read_batch()

    def release(self) -> None:
        """Release Resources
        """
        if self._is_open is True:
            self._frame_count = self._num_files
    
    def show(self, frame: np.ndarray | None) -> None:
        """Show video
        """
        cv2.imshow(self._name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.release()
            cv2.destroyAllWindows()
            print("Exiting...")

    def __del__(self) -> None:
        """Release Resources
        """

    def __next__(self) -> np.ndarray:
        """Returns next frame from the video
        Raises:
            StopIteration: No more frames to read
        Returns:
            np.ndarray: frame read from video
        """
        if self._frame_count == self._num_files:
            raise StopIteration
        frame = self.read()

        return frame

    def __iter__(self) -> "ReaderInterface":
        """Returns iterable object for reading frames
        Returns:
            Iterable[ReaderInterface]: iterable object for reading frames
        """
        self._frame_count = 0
        self.batch = 0
        return self

    def __repr__(self) -> str:
        """Video's Info
        Returns:
            str: info
        """
        return str(self._info)

    def __str__(self) -> str:
        """Video's Info
        Returns:
            str: Info
        """
        return str(self._info)

    def __enter__(self) -> "ReaderInterface":
        """Returns Conext for "with" block usage
        Returns:
            ReaderInterface: Video Reader object
        """
        return self

    def __exit__(self, exc_type: None, exc_value: None,
                 traceback: None) -> None:
        """Release resources before exiting the "with" block
        Args:
            exc_type (NoneType): Exception type if any
            exc_value (NoneType): Exception value if any
            traceback (NoneType): Traceback of Exception
        """