# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
from unittest import mock
import torch

from detectron2.modeling import poolers
from detectron2.modeling.proposal_generator import rpn
from detectron2.modeling.roi_heads import keypoint_head, mask_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from .c10 import (
    Caffe2Compatible,
    Caffe2FastRCNNOutputsInference,
    Caffe2KeypointRCNNInference,
    Caffe2MaskRCNNInference,
    Caffe2ROIPooler,
    Caffe2RPN,
)


class GenericMixin(object):
    pass


class Caffe2CompatibleConverter(object):
    """
    A GenericUpdater which implements the `create_from` interface, by modifying
    module object and assign it with another class replaceCls.
    """

    def __init__(self, replaceCls):
        self.replaceCls = replaceCls

    def create_from(self, module):
        # update module's class to the new class
        assert isinstance(module, torch.nn.Module)
        if issubclass(self.replaceCls, GenericMixin):
            # replaceCls should act as mixin, create a new class on-the-fly
            new_class = type(
                "{}MixedWith{}".format(self.replaceCls.__name__, module.__class__.__name__),
                (self.replaceCls, module.__class__),
                {},  # {"new_method": lambda self: ...},
            )
            module.__class__ = new_class
        else:
            # replaceCls is complete class, this allow arbitrary class swap
            module.__class__ = self.replaceCls

        # initialize Caffe2Compatible
        if isinstance(module, Caffe2Compatible):
            module.tensor_mode = False

        return module






@contextlib.contextmanager


@contextlib.contextmanager


@contextlib.contextmanager



def patch(model, target, updater, *args, **kwargs):
    """
    recursively (post-order) update all modules with the target type and its
    subclasses, make a initialization/composition/inheritance/... via the
    updater.create_from.
    """
    for name, module in model.named_children():
        model._modules[name] = patch(module, target, updater, *args, **kwargs)
    if isinstance(model, target):
        return updater.create_from(model, *args, **kwargs)
    return model


def patch_generalized_rcnn(model):
    ccc = Caffe2CompatibleConverter
    model = patch(model, rpn.RPN, ccc(Caffe2RPN))
    model = patch(model, poolers.ROIPooler, ccc(Caffe2ROIPooler))

    return model


@contextlib.contextmanager
def mock_fastrcnn_outputs_inference(
    tensor_mode, check=True, box_predictor_type=FastRCNNOutputLayers
):
    with mock.patch.object(
        box_predictor_type,
        "inference",
        autospec=True,
        side_effect=Caffe2FastRCNNOutputsInference(tensor_mode),
    ) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0


@contextlib.contextmanager
def mock_mask_rcnn_inference(tensor_mode, patched_module, check=True):
    with mock.patch(
        "{}.mask_rcnn_inference".format(patched_module), side_effect=Caffe2MaskRCNNInference()
    ) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0


@contextlib.contextmanager
def mock_keypoint_rcnn_inference(tensor_mode, patched_module, use_heatmap_max_keypoint, check=True):
    with mock.patch(
        "{}.keypoint_rcnn_inference".format(patched_module),
        side_effect=Caffe2KeypointRCNNInference(use_heatmap_max_keypoint),
    ) as mocked_func:
        yield
    if check:
        assert mocked_func.call_count > 0


class ROIHeadsPatcher:

    @contextlib.contextmanager