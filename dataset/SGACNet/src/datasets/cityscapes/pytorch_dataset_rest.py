# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2

from ..dataset_base import DatasetBase
from .cityscapes import CityscapesBase


class Cityscapes(CityscapesBase, DatasetBase):

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property






            self._files = {
                'rgb': _loadtxt(f'{self._split}_rgb.txt'),
                self._depth_dir: _loadtxt(f'{self._split}_{self._depth_dir}.txt'),
                'label': _loadtxt(f'{self._split}_labels_{self._n_classes}.txt'),
            }
            assert all(len(l) == len(self._files['rgb'])
                       for l in self._files.values())
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        # class names, class colors, and label directory
        if self._n_classes == 19:
            self._class_names = self.CLASS_NAMES_REDUCED
            self._class_colors = np.array(self.CLASS_COLORS_REDUCED,
                                          dtype='uint8')
            self._label_dir = self.LABELS_REDUCED_DIR
        else:
            self._class_names = self.CLASS_NAMES_FULL
            self._class_colors = np.array(self.CLASS_COLORS_FULL,
                                          dtype='uint8')
            self._label_dir = self.LABELS_FULL_DIR

        if disparity_instead_of_depth:
            self._depth_mean = 9069.706336834102
            self._depth_std = 7178.335960071306
        else:
            self._depth_mean = 31.715617493177906
            self._depth_std = 38.70280704877372

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, directory, filename):
        fp = os.path.join(self._data_dir,
                          self.split,
                          directory,
                          filename)
        if os.path.splitext(fp)[-1] == '.npy':
            # depth files as numpy files
            return np.load(fp)
        else:
            # all the other files are pngs
            im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._files['rgb'][idx])

    def load_depth(self, idx):
        depth = self._load(self._depth_dir,
                           self._files[self._depth_dir][idx])
        if depth.dtype == 'float16':
            # precomputed depth values are stored as float16 -> cast to float32
            depth = depth.astype('float32')
            # set values larger than 300 to zero as they are most likley not
            # valid
            depth[depth > 300] = 0
        return depth

    def load_label(self, idx):
        return self._load(self._label_dir, self._files['label'][idx])

    def __len__(self):
        return len(self._files['rgb'])