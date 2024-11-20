def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True,
    _verbose=True, device=None):
    return _create('yolov5l', pretrained, channels, classes, autoshape,
        _verbose, device)
