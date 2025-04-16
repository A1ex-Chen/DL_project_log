# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)










if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("gt_dir")
    parser.add_argument("--type", choices=["instance", "semantic"], default="instance")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer
    from cityscapesscripts.helpers.labels import labels

    logger = setup_logger(name=__name__)

    dirname = "cityscapes-data-vis"
    os.makedirs(dirname, exist_ok=True)

    if args.type == "instance":
        dicts = load_cityscapes_instances(
            args.image_dir, args.gt_dir, from_json=True, to_polygons=True
        )
        logger.info("Done loading {} samples.".format(len(dicts)))

        thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
        meta = Metadata().set(thing_classes=thing_classes)

    else:
        dicts = load_cityscapes_semantic(args.image_dir, args.gt_dir)
        logger.info("Done loading {} samples.".format(len(dicts)))

        stuff_classes = [k.name for k in labels if k.trainId != 255]
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)

    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)