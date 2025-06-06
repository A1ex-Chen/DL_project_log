# Ultralytics YOLO 🚀, AGPL-3.0 license

import types
from copy import deepcopy

import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, callbacks
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.model import YOLO

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP






    obj.model.loss = types.MethodType(loss, obj.model)
    obj.model.forward = types.MethodType(forward, obj.model)

    obj.model.names = [''] if not num_classes else [f'class{i}' for i in range(num_classes)]
    obj.overrides['model'] = obj.cfg

    # Below added to allow export from yamls
    args = {**DEFAULT_CFG_DICT, **obj.overrides}  # combine model and default args, preferring model args
    obj.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    obj.model.task = obj.task
    obj.model.model_name = model_name


def patched_export(obj, model_name='model', **kwargs):
    obj.model.yaml = {'yaml_file': model_name}

    obj._check_is_pytorch_model()
    overrides = obj.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'export'
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    args.task = obj.task
    if args.imgsz == DEFAULT_CFG.imgsz:
        args.imgsz = obj.model.args['imgsz']  # use trained imgsz unless custom value is passed
    if args.batch == DEFAULT_CFG.batch:
        args.batch = 1  # default to 1 if not modified

    # Update model
    model = deepcopy(obj.model).to(obj.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, tuple(HEAD_NAME_MAP.values())):
            m.dynamic = args.dynamic
            m.export = True
            m.format = args.format

    model.yaml_file = model_name
    return Exporter(overrides=args, _callbacks=obj.callbacks)(model=model)


YOLO.__init__ = patched_init
YOLO.export = patched_export