from __future__ import annotations

import numpy as np

class Det:
    def __init__(self, filter_cls: list[int], pred: np.ndarray):
        self.filter_cls = filter_cls
        filtered_idx = np.in1d(pred[:, 5], filter_cls)

        if len(filtered_idx) > 0 :
            self.pred = pred[filtered_idx]
        else:
            self.pred = np.array([])

        self._xyxy = None
        self._cls = None
        self._conf = None
        self._tlwh = None
        self._tlbr = None
        self._cxywh = None
    
    @property
    def xyxy(self):
        if self._xyxy is None:
            self._xyxy = self.pred[:, 0:4]
        return self._xyxy
    
    @property
    def cls(self):
        if self._cls is None:
            self._cls = self.pred[:, 5]
        return self._cls
    
    @property
    def conf(self):
        if self._conf is None:
            self._conf = self.pred[:, 4]
        return self._conf

    @property
    def tlwh(self):
        if self._tlwh is None:
            ret = self.xyxy.copy()
            ret[:2] -= ret[2:] / 2
            self._tlwh = ret
        return self._tlwh
    
    @property
    def tlbr(self):
        if self._tlbr is None:
            ret = self.tlwh.copy()
            ret[2:] += ret[:2]
            self._tlbr = ret
        return self._tlbr
    
    @property
    def cxywh(self):
        if self._cxywh is None:
            ret = self.xyxy.copy()
            ret[:, :2] = ret[:, :2] + ret[:, 2:] / 2
            ret[:, 2:] -= ret[:, :2]
            self._cxywh = ret
        return self._cxywh
