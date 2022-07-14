from __future__ import annotations
import os
import json

import numpy as np

from inference.utils.sorting import get_sorted_alpanumeric_files
from .data_class import Box, Person

class DavidDataset:
    def __init__(self, path: str, padding_size: tuple[int, int] | list[int, int] | None = None):
        self._init_props()
        self._post_init(path, padding_size)

    def _init_props(self):
        self._files = None
        self._num_files = None
        self._padding_size = None
        self._info = None
        self._name = None
        self._data = None
        self._frame_count = 0

    def _post_init(self, path: str, padding_size: tuple[int, int] | list[int, int] | None = None):
        """Update info property
        """
        self._files = get_sorted_alpanumeric_files(path, ['json'])
        self._num_files = len(self._files)
        self._padding_size = (0,0) if padding_size is None else padding_size
        self._info = {
            "name": None,
            "frame_count": 0,
            "padding_size": self._padding_size,
            "num_files": self._num_files,
        }
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def info(self) -> dict[str, str]:
        return self._info
    
    @property
    def data(self) -> dict[str, str]:
        return self._data

    def load_data(self):
        with open(self._files[self._frame_count]) as f:
            data = json.load(f)
        self._frame_count += 1
        self._data = data
    
    def aggregate(self):
        def get_box_output(rows: dict[str,str], filed_name: str) -> Box:
            box = rows.get(filed_name, {}).get('box')
            if box:
                xywh = (box['x'] - self._padding_size[0],
                        box['y'] - self._padding_size[1], 
                        box['width'],
                        box['height'])
                return Box(*xywh)
            return Box(0,0,0,0)
            
        aggregated = {'name': self.name, 'frame': self.frame_count}
        aggregated['data'] = []
        
        for rows in self._data:
            track_id = rows.get('tracking_id')
            if track_id:
                track_id = int(track_id)
                visible = get_box_output(rows, 'visible-box')
                head = get_box_output(rows, 'head-box')
                full = get_box_output(rows, 'full-box')

                body_key_points = {k:(v['x'],v['y']) for k,v in rows.get('body-key-points', {}).items()}
                
                aggregated['data'].append(Person(track_id, visible, full, head, body_key_points))
            
        self._data = aggregated

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index: int) -> dict[str, str]:
        with open(self._files[index]) as f:
            data = json.load(f)
        return data
    
    def __repr__(self):
        return f'<Dataset: {self.name}>'
    
    def __str__(self):
        return self.name

    def __iter__(self):
        self._frame_count = 0
        return self

    def __next__(self) -> dict[str, str]:
        if self._frame_count >= self._num_files:
            raise StopIteration
        self.load_data()
        self.aggregate()
        return self.data['data']
