def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True,
    _verbose=True, device=None):
    return _create('yolov5x', pretrained, channels, classes, autoshape,
        _verbose, device)
