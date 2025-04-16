from typing import Any, Dict, List
import json
import h5py
import numpy as np
from tqdm import tqdm
import random
from anytree import AnyNode
from anytree.search import findall_by_attr,findall
import copy
import string

class OIDictImporter(object):
    ''' Importer that works on Open Images json hierarchy '''
    def __init__(self, nodecls=AnyNode):
        self.nodecls = nodecls

    def import_(self, data):
        """Import tree from `data`."""
        return self.__import(data)


    def __import(self, data, parent=None):
        assert isinstance(data, dict)
        assert "parent" not in data
        attrs = dict(data)
        children = attrs.pop("Subcategory", [])
        node = self.nodecls(parent=parent, **attrs)
        for child in children:
            self.__import(child, parent=node)
        return node

class HierarchyFinder(object):

    def __init__(self, class_structure_path, abstract_list_path):
        importer = OIDictImporter()
        with open(class_structure_path) as f:
            self.class_structure = importer.import_(json.load(f))

        with open(abstract_list_path) as out:
            self.abstract_list = json.load(out)

    def find_key(self, label):
        if label in self.abstract_list:
            return label
        return None

    def find_parent(self, label):
        target_node = findall(self.class_structure, filter_=lambda node: node.LabelName.lower() in (label))[0]
        while  self.find_key(target_node.LabelName.lower()) is None:
            target_node = target_node.parent
        return self.find_key(target_node.LabelName.lower())





class HierarchyFinder(object):




def nms(dets, classes, hierarchy, thresh=0.8):
    # Non-max suppression of overlapping boxes where score is based on 'height' in the hierarchy,
    # defined as the number of edges on the longest path to a leaf
    scores = [findall(hierarchy, filter_=lambda node: node.LabelName.lower() == cls)[0].height for cls in classes]
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = np.array(scores)
    order = scores.argsort()

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # check the score, objects with smaller or equal number of layers cannot be removed.
        keep_condition = np.logical_or(scores[order[1:]] <= scores[i], \
            inter / (areas[i] + areas[order[1:]] - inter) <= thresh)

        inds = np.where(keep_condition)[0]
        order = order[inds + 1]

    return keep

class ImageFeaturesReader(object):
    r"""
    A reader for H5 files containing pre-extracted image features. A typical image features file
    should have at least two H5 datasets, named ``image_id`` and ``features``. It may optionally
    have other H5 datasets, such as ``boxes`` (for bounding box coordinates), ``width`` and
    ``height`` for image size, and others. This reader only reads image features, because our
    UpDown captioner baseline does not require anything other than image features.

    Example of an h5 file::

        image_bottomup_features.h5
        |--- "image_id" [shape: (num_images, )]
        |--- "features" [shape: (num_images, num_boxes, feature_size)]
        +--- .attrs {"split": "coco_train2017"}

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing image ids and features corresponding to one of the four
        ``split``s used: "coco_train2017", "coco_val2017", "nocaps_val", "nocaps_test".
    in_memory : bool
        Whether to load the features in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM.
    """







class CocoCaptionsReader(object):





class BoxesReader(object):
    """
    A reader for H5 files containing bounding boxes, classes and confidence scores inferred using
    an object detector. A typical H5 file should at least have the following structure:
    ```
    image_boxes.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "width" [shape: (num_images, )]
       |--- "height" [shape: (num_images, )]
       |--- "boxes" [shape: (num_images, max_num_boxes, 4)]
       |--- "classes" [shape: (num_images, max_num_boxes, )]
       +--- "scores" [shape: (num_images, max_num_boxes, )]
    ```
    Box coordinates are of form [X1, Y1, X2, Y2], _not_ normalized by image width and height. Class
    IDs start from 1, i-th ID corresponds to (i-1)-th category in "categories" field of
    corresponding annotation file for this split (in COCO format).
    Parameters
    ----------
    boxes_h5path : str
        Path to an H5 file containing boxes, classes and scores of a particular dataset split.
    """

        



