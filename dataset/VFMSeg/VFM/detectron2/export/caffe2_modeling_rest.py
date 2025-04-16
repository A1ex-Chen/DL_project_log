# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import io
import struct
import types
import torch

from detectron2.modeling import meta_arch
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads import keypoint_head
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes

from .c10 import Caffe2Compatible
from .caffe2_patch import ROIHeadsPatcher, patch_generalized_rcnn
from .shared import (
    alias,
    check_set_pb_arg,
    get_pb_arg_floats,
    get_pb_arg_valf,
    get_pb_arg_vali,
    get_pb_arg_vals,
    mock_torch_nn_functional_interpolate,
)









    model.apply(_fn)


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    See get_caffe2_inputs() below.
    """
    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # NOTE: The scale inside im_info is kept as convention and for providing
        # post-processing information if further processing is needed. For
        # current Caffe2 model definitions that don't include post-processing inside
        # the model, this number is not used.
        # NOTE: There can be a slight difference between width and height
        # scales, using a single number can results in numerical difference
        # compared with D2's post-processing.
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)


class Caffe2MetaArch(Caffe2Compatible, torch.nn.Module):
    """
    Base class for caffe2-compatible implementation of a meta architecture.
    The forward is traceable and its traced graph can be converted to caffe2
    graph through ONNX.
    """






    @staticmethod


class Caffe2GeneralizedRCNN(Caffe2MetaArch):


    @mock_torch_nn_functional_interpolate()

    @staticmethod


class Caffe2RetinaNet(Caffe2MetaArch):

    @mock_torch_nn_functional_interpolate()



    @staticmethod

        return f


class Caffe2RetinaNet(Caffe2MetaArch):
    def __init__(self, cfg, torch_model):
        assert isinstance(torch_model, meta_arch.RetinaNet)
        super().__init__(cfg, torch_model)

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        assert self.tensor_mode
        images = self._caffe2_preprocess_image(inputs)

        # explicitly return the images sizes to avoid removing "im_info" by ONNX
        # since it's not used in the forward path
        return_tensors = [images.image_sizes]

        features = self._wrapped_model.backbone(images.tensor)
        features = [features[f] for f in self._wrapped_model.head_in_features]
        for i, feature_i in enumerate(features):
            features[i] = alias(feature_i, "feature_{}".format(i), is_backward=True)
            return_tensors.append(features[i])

        pred_logits, pred_anchor_deltas = self._wrapped_model.head(features)
        for i, (box_cls_i, box_delta_i) in enumerate(zip(pred_logits, pred_anchor_deltas)):
            return_tensors.append(alias(box_cls_i, "box_cls_{}".format(i)))
            return_tensors.append(alias(box_delta_i, "box_delta_{}".format(i)))

        return tuple(return_tensors)

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(
            predict_net, "device", "s", str.encode(str(self._wrapped_model.device), "ascii")
        )
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"RetinaNet")

        # Inference parameters:
        check_set_pb_arg(
            predict_net, "score_threshold", "f", _cast_to_f32(self._wrapped_model.test_score_thresh)
        )
        check_set_pb_arg(
            predict_net, "topk_candidates", "i", self._wrapped_model.test_topk_candidates
        )
        check_set_pb_arg(
            predict_net, "nms_threshold", "f", _cast_to_f32(self._wrapped_model.test_nms_thresh)
        )
        check_set_pb_arg(
            predict_net,
            "max_detections_per_image",
            "i",
            self._wrapped_model.max_detections_per_image,
        )

        check_set_pb_arg(
            predict_net,
            "bbox_reg_weights",
            "floats",
            [_cast_to_f32(w) for w in self._wrapped_model.box2box_transform.weights],
        )
        self._encode_anchor_generator_cfg(predict_net)

    def _encode_anchor_generator_cfg(self, predict_net):
        # serialize anchor_generator for future use
        serialized_anchor_generator = io.BytesIO()
        torch.save(self._wrapped_model.anchor_generator, serialized_anchor_generator)
        # Ideally we can put anchor generating inside the model, then we don't
        # need to store this information.
        bytes = serialized_anchor_generator.getvalue()
        check_set_pb_arg(predict_net, "serialized_anchor_generator", "s", bytes)

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        self = types.SimpleNamespace()
        serialized_anchor_generator = io.BytesIO(
            get_pb_arg_vals(predict_net, "serialized_anchor_generator", None)
        )
        self.anchor_generator = torch.load(serialized_anchor_generator)
        bbox_reg_weights = get_pb_arg_floats(predict_net, "bbox_reg_weights", None)
        self.box2box_transform = Box2BoxTransform(weights=tuple(bbox_reg_weights))
        self.test_score_thresh = get_pb_arg_valf(predict_net, "score_threshold", None)
        self.test_topk_candidates = get_pb_arg_vali(predict_net, "topk_candidates", None)
        self.test_nms_thresh = get_pb_arg_valf(predict_net, "nms_threshold", None)
        self.max_detections_per_image = get_pb_arg_vali(
            predict_net, "max_detections_per_image", None
        )

        # hack to reuse inference code from RetinaNet
        for meth in [
            "forward_inference",
            "inference_single_image",
            "_transpose_dense_predictions",
            "_decode_multi_level_predictions",
            "_decode_per_level_predictions",
        ]:
            setattr(self, meth, functools.partial(getattr(meta_arch.RetinaNet, meth), self))


        return f


META_ARCH_CAFFE2_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": Caffe2GeneralizedRCNN,
    "RetinaNet": Caffe2RetinaNet,
}