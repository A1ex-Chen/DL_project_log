# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2

from .sunrgbd import SUNRBDBase
from ..dataset_base import DatasetBase


class SUNRGBD(SUNRBDBase, DatasetBase):

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







        img_dir_train_file = _get_filepath('train_rgb.txt')
        depth_dir_train_file = _get_filepath('train_depth.txt')
        label_dir_train_file = _get_filepath('train_label.txt')

        img_dir_test_file = _get_filepath('test_rgb.txt')
        depth_dir_test_file = _get_filepath('test_depth.txt')
        label_dir_test_file = _get_filepath('test_label.txt')

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in ['train', 'test']:
            img_dir[phase] = dict()
            depth_dir[phase] = dict()
            label_dir[phase] = dict()

        img_dir['train']['list'], img_dir['train']['dict'] = \
            self.list_and_dict_from_file(img_dir_train_file)
        depth_dir['train']['list'], depth_dir['train']['dict'] = \
            self.list_and_dict_from_file(depth_dir_train_file)
        label_dir['train']['list'], label_dir['train']['dict'] = \
            self.list_and_dict_from_file(label_dir_train_file)

        img_dir['test']['list'], img_dir['test']['dict'] = \
            self.list_and_dict_from_file(img_dir_test_file)
        depth_dir['test']['list'], depth_dir['test']['dict'] = \
            self.list_and_dict_from_file(depth_dir_test_file)
        label_dir['test']['list'], label_dir['test']['dict'] = \
            self.list_and_dict_from_file(label_dir_test_file)

        return img_dir, depth_dir, label_dir

    def list_and_dict_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        dictionary = dict()
        for cam in self.cameras:
            dictionary[cam] = [i for i in file_list if cam in i]

        return file_list, dictionary

    def __len__(self):
        if self.camera is None:
            return len(self.img_dir[self._split]['list'])
        return len(self.img_dir[self._split]['dict'][self.camera])