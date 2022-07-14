from __future__ import annotations
import os

import cv2
import json
import numpy as np
from PIL import Image

from .plot import draw_xyxy_box, draw_key_point
from inference.interface.reader import ReaderInterface
from inference.interface.writer import WriterInterface


class Writer(WriterInterface):
    """
    Writer class for writing data to a destination.
    """

    _EXT_TO_FOURCC = {
        ".avi": "DIVX",
        ".mkv": "X264",
        ".mp4": "mp4v",
        ".mov": "mov",
        ".wmv": "WMV2",
        ".webm": "VP80",
        ".flv": "FLV1",
        ".mpg": "MP42",
    }

    def __init__(self,
                 reader: ReaderInterface | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 fps: float | None = None,
                 name: str | None = None,
                 ext: str | None = None,
                 output_dir: str | None = None,
                 **kwargs) -> None:

        self._init_props()

        self._update_props(reader, width, height, fps, name, ext, output_dir,
                           **kwargs)

        self._video_writer = cv2.VideoWriter(self._video_file_name,
                                             self._fourcc(), self._fps,
                                             (self._width, self._height))

        self._image_writer = cv2.imwrite

        self._update_info()

    def _init_props(self) -> None:
        """
        Initialize properties.
        """
        self._name = None
        self._video_writer = None
        self._width = None
        self._height = None
        self._fps = None
        self._ext = None
        self._info = None
        self._frame_count = 0
        self._seconds = 0
        self._minutes = 0

    def _update_props(self,
                      reader: ReaderInterface | None = None,
                      width: int | None = None,
                      height: int | None = None,
                      fps: float | None = None,
                      name: str | None = None,
                      ext: str | None = None,
                      output_dir: str | None = None,
                      **kwargs) -> None:
        """
        Update properties.
        """
        if reader is not None:
            self._name = reader.name
            self._width = reader.width
            self._height = reader.height
            self._fps = reader.fps
            self._info = reader.info
            self._frame_count = reader.frame_count
            self._seconds = reader.seconds
            self._minutes = reader.minutes

        if name is not None:
            self._name = name
        self._name = "Webcam.mp4" if self._name.isdecimal() else self._name
        self._base_name = os.path.basename(self._name).split(".")[0]

        if width is not None:
            self._width = width

        if height is not None:
            self._height = height

        if fps is not None:
            self._fps = fps

        # check valid name
        if self._name is None:
            raise AssertionError("Must provide either Reader or name arg.")

        # set ext for CV2 writer creations
        self._ext = '.mp4' if ext is None else ext

        if '.' not in self._ext:
            self._ext = '.' + self._ext

        if self._ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
            self._ext = '.mp4'

        if output_dir is None:
            output_dir = 'output/'

        self._image_output_dir = os.path.join(output_dir, self._base_name)
        self._video_file_name = os.path.join(output_dir, f'{self._base_name}{self._ext}')
        self._result_file_name = os.path.join(output_dir, f'{self._base_name}_result')

        # make output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(self._image_output_dir):
            os.makedirs(self._image_output_dir)

    def _update_info(self) -> None:
        """Update info property according to class props
        """
        # update info
        self._info = {
            "name": self._name,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "ext": self._ext
        }

    def _fourcc(self) -> cv2.VideoWriter_fourcc:
        return cv2.VideoWriter_fourcc(*self._EXT_TO_FOURCC.get(self._ext, 'mp4v'))

    @property
    def name(self) -> str:
        """Name of Output Video
        Returns:
            str: name of output video
        """
        return self._name

    @property
    def width(self) -> int:
        """Width of Output Video
        Returns:
            int: width of video frame
        """
        return self._width

    @property
    def height(self) -> int:
        """Height of Output Video
        Returns:
            int: height of video frame
        """
        return self._height

    @property
    def fps(self) -> float:
        """FPS of Output Video
        Returns:
            float: fps of video
        """
        return self._fps

    @property
    def ext(self) -> str:
        """Extension of Output Video
        Returns:
            str: ext of video
        """
        return self._ext

    @property
    def info(self) -> dict:
        """Video information
        Returns:
            dict: info of width, height, fps, backend and ext
        """
        return self._info

    @property
    def frame_count(self) -> int:
        """Total frames written
        Returns:
            int: written frames' count
        """
        return self._frame_count

    @property
    def seconds(self) -> float:
        """Total seconds written
        Returns:
            float: written frames' in seconds
        """
        return (self._frame_count / self._fps) if self._fps else 0

    @property
    def minutes(self) -> float:
        """Total minutes written
        Returns:
            float: written frames' in minutes
        """
        return self.seconds / 60.0

    def is_open(self) -> bool:
        """Checks if writer is still open
        Returns:
            bool: True if writer is open, False otherwise
        """
        return self._video_writer.isOpened()

    def write_vid(self, frame: np.ndarray) -> None:
        """Write frame to output video
        Args:
            frame (np.ndarray): frame to write
        Raises:
            Exception: raised when method is called on a non-open writer.
        """
        # check if writer is open
        if not self.is_open():
            raise Exception("Attempted writing with a non-open Writer.")

        self._video_writer.write(frame)
        self._frame_count += 1

    def write_img(self, frame: np.ndarray) -> None:
        """Write frame to output image
        Args:
            frame (np.ndarray): frame to write
        """

        # zero fill frame number
        file_name = f'{self._image_output_dir}/{str(self._frame_count).zfill(6)}.jpg'
        self._image_writer(file_name, frame)

    def release(self) -> None:
        """Release Resources
        """
        if self._video_writer is not None:
            self._video_writer.release()

    def draw_bbox(self, image: Image, xyxy: tuple[str | int], label: str, box_id: int) -> None:
        """Draw bounding box on output video
        """
        draw_xyxy_box(image, xyxy, label, box_id)
    
    def draw_key_points(self, image: Image, key_point: dict[str, tuple[int, int]], size: int=5) -> None:
        for point_id, value in key_point.items():
            draw_key_point(image, point_id, value, size)
    
    def save_txt(self, outputs: list[str]) -> None:
        """Save bounding box on output video
        """
        file_name = f'{self._image_output_dir}/{str(self._frame_count).zfill(6)}.txt'
        with open(file_name, 'w') as f:
            for output in outputs:
                f.write(output + '\n')
    
    def save_json(self, outputs: list[str]) -> None:
        """Save bounding box on output video
        """
        pass


    def __del__(self) -> None:
        """Release Resources
        """
        self.release()
        self._video_writer = None

    def __repr__(self) -> str:
        """Writer's Info
        Returns:
            str: info
        """
        return str(self.info)

    def __str__(self) -> str:
        """Writer's Info
        Returns:
            str: info
        """
        return str(self.info)

    def __enter__(self) -> "Writer":
        """Returns Conext for "with" block usage
        Returns:
            Writer: Video Reader object
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
        self.release()