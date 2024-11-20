import pathlib
import pickle
import time
from functools import partial

import numpy as np

from tDBN.core import box_np_ops
from tDBN.core import preprocess as prep
from tDBN.kitti import kitti_common as kitti
from tDBN.kitti.preprocess import _read_and_prep_v9


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """





class KittiDataset(Dataset):


    @property
