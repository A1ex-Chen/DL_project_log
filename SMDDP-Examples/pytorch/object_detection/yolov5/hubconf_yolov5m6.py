def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True,
    _verbose=True, device=None):
    return _create('yolov5m6', pretrained, channels, classes, autoshape,
        _verbose, device)
