def _create(name, pretrained=True, channels=3, classes=80, autoshape=True,
    verbose=True, device=None):
    """Creates or loads a YOLOv5 model

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path
    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import Model
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device
    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir(
        ) else name
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)
                if autoshape:
                    model = AutoShape(model)
            except Exception:
                model = attempt_load(path, device=device, fuse=False)
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(
                f'{path.stem}.yaml'))[0]
            model = Model(cfg, channels, classes)
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
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = (
            f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
            )
        raise Exception(s) from e
