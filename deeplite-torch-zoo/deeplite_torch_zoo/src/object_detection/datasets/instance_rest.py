# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np

from deeplite_torch_zoo.src.object_detection.datasets.utils import ltwh2xywh, ltwh2xyxy, resample_segments, \
    xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(yolo format)
# `ltwh` means left top and width, height(coco format)
_formats = ['xyxy', 'xywh', 'ltwh']



    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)


class Bboxes:
    """Now only numpy is supported."""

        # self.normalized = normalized

    # def convert(self, format):
    #     assert format in _formats
    #     if self.format == format:
    #         bboxes = self.bboxes
    #     elif self.format == "xyxy":
    #         if format == "xywh":
    #             bboxes = xyxy2xywh(self.bboxes)
    #         else:
    #             bboxes = xyxy2ltwh(self.bboxes)
    #     elif self.format == "xywh":
    #         if format == "xyxy":
    #             bboxes = xywh2xyxy(self.bboxes)
    #         else:
    #             bboxes = xywh2ltwh(self.bboxes)
    #     else:
    #         if format == "xyxy":
    #             bboxes = ltwh2xyxy(self.bboxes)
    #         else:
    #             bboxes = ltwh2xywh(self.bboxes)
    #
    #     return Bboxes(bboxes, format)



    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True




    @classmethod



class Instances:



    @property












    @classmethod

    @property