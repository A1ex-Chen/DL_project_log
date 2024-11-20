# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import types
from collections import UserDict
from typing import List

from detectron2.utils.logger import log_first_n

__all__ = ["DatasetCatalog", "MetadataCatalog", "Metadata"]


class _DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """






    __repr__ = __str__


DatasetCatalog = _DatasetCatalog()
DatasetCatalog.__doc__ = (
    _DatasetCatalog.__doc__
    + """
    .. automethod:: detectron2.data.catalog.DatasetCatalog.register
    .. automethod:: detectron2.data.catalog.DatasetCatalog.get
"""
)


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:
    ::
        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }







class _MetadataCatalog(UserDict):
    """
    MetadataCatalog is a global dictionary that provides access to
    :class:`Metadata` of a given dataset.

    The metadata associated with a certain name is a singleton: once created, the
    metadata will stay alive and will be returned by future calls to ``get(name)``.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    """





    __repr__ = __str__


MetadataCatalog = _MetadataCatalog()
MetadataCatalog.__doc__ = (
    _MetadataCatalog.__doc__
    + """
    .. automethod:: detectron2.data.catalog.MetadataCatalog.get
"""
)