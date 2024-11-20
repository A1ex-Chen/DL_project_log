def _create(name, pretrained=True, channels=3, classes=80, autoshape=True,
    verbose=True, device=None):
    """
    Creates or loads a YOLOv3 model.

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv3 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv3 model
    """
    from pathlib import Path
    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device
    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / 'requirements.txt', exclude=('opencv-python',
        'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir(
        ) else name
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel
                        ):
                        LOGGER.warning(
                            'WARNING ⚠️ YOLOv3 ClassificationModel is not yet AutoShape compatible. You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).'
                            )
                    elif model.pt and isinstance(model.model, SegmentationModel
                        ):
                        LOGGER.warning(
                            'WARNING ⚠️ YOLOv3 SegmentationModel is not yet AutoShape compatible. You will not be able to run inference with this model.'
                            )
                    else:
                        model = AutoShape(model)
            except Exception:
                model = attempt_load(path, device=device, fuse=False)
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(
                f'{path.stem}.yaml'))[0]
            model = DetectionModel(cfg, channels, classes)
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=[
                    'anchors'])
                model.load_state_dict(csd, strict=False)
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names
        if not verbose:
            LOGGER.setLevel(logging.INFO)
        return model.to(device)
    except Exception as e:
        help_url = (
            'https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading'
            )
        s = (
            f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
            )
        raise Exception(s) from e
