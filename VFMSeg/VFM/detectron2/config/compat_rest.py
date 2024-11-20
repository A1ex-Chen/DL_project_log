# Copyright (c) Facebook, Inc. and its affiliates.
"""
Backward compatibility of configs.

Instructions to bump version:
+ It's not needed to bump version if new keys are added.
  It's only needed when backward-incompatible changes happen
  (i.e., some existing keys disappear, or the meaning of a key changes)
+ To bump version, do the following:
    1. Increment _C.VERSION in defaults.py
    2. Add a converter in this file.

      Each ConverterVX has a function "upgrade" which in-place upgrades config from X-1 to X,
      and a function "downgrade" which in-place downgrades config from X to X-1

      In each function, VERSION is left unchanged.

      Each converter assumes that its input has the relevant keys
      (i.e., the input is not a partial config).
    3. Run the tests (test_config.py) to make sure the upgrade & downgrade
       functions are consistent.
"""

import logging
from typing import List, Optional, Tuple

from .config import CfgNode as CN
from .defaults import _C

__all__ = ["upgrade_config", "downgrade_config"]









    # Most users' partial configs have "MODEL.WEIGHT", so guess on it
    ret = None
    if _has("MODEL.WEIGHT") or _has("TEST.AUG_ON"):
        ret = 1

    if ret is not None:
        logger.warning("Config '{}' has no VERSION. Assuming it to be v{}.".format(filename, ret))
    else:
        ret = _C.VERSION
        logger.warning(
            "Config '{}' has no VERSION. Assuming it to be compatible with latest v{}.".format(
                filename, ret
            )
        )
    return ret


def _rename(cfg: CN, old: str, new: str) -> None:
    old_keys = old.split(".")
    new_keys = new.split(".")




    _set(new_keys, _get(old_keys))
    _del(old_keys)


class _RenameConverter:
    """
    A converter that handles simple rename.
    """

    RENAME: List[Tuple[str, str]] = []  # list of tuples of (old name, new name)

    @classmethod

    @classmethod


class ConverterV1(_RenameConverter):
    RENAME = [("MODEL.RPN_HEAD.NAME", "MODEL.RPN.HEAD_NAME")]


class ConverterV2(_RenameConverter):
    """
    A large bulk of rename, before public release.
    """

    RENAME = [
        ("MODEL.WEIGHT", "MODEL.WEIGHTS"),
        ("MODEL.PANOPTIC_FPN.SEMANTIC_LOSS_SCALE", "MODEL.SEM_SEG_HEAD.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.RPN_LOSS_SCALE", "MODEL.RPN.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.INSTANCE_LOSS_SCALE", "MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.COMBINE_ON", "MODEL.PANOPTIC_FPN.COMBINE.ENABLED"),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_OVERLAP_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_STUFF_AREA_LIMIT",
            "MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_INSTANCES_CONFIDENCE_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH",
        ),
        ("MODEL.ROI_HEADS.SCORE_THRESH", "MODEL.ROI_HEADS.SCORE_THRESH_TEST"),
        ("MODEL.ROI_HEADS.NMS", "MODEL.ROI_HEADS.NMS_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_SCORE_THRESHOLD", "MODEL.RETINANET.SCORE_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_TOPK_CANDIDATES", "MODEL.RETINANET.TOPK_CANDIDATES_TEST"),
        ("MODEL.RETINANET.INFERENCE_NMS_THRESHOLD", "MODEL.RETINANET.NMS_THRESH_TEST"),
        ("TEST.DETECTIONS_PER_IMG", "TEST.DETECTIONS_PER_IMAGE"),
        ("TEST.AUG_ON", "TEST.AUG.ENABLED"),
        ("TEST.AUG_MIN_SIZES", "TEST.AUG.MIN_SIZES"),
        ("TEST.AUG_MAX_SIZE", "TEST.AUG.MAX_SIZE"),
        ("TEST.AUG_FLIP", "TEST.AUG.FLIP"),
    ]

    @classmethod

    @classmethod