def check_amp(model):
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        m = AutoShape(model, verbose=False)
        a = m(im).xywhn[0]
        m.amp = True
        b = m(im).xywhn[0]
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)
    prefix = colorstr('AMP: ')
    device = next(model.parameters()).device
    if device.type == 'cpu':
        return False
    f = ROOT / 'data' / 'images' / 'bus.jpg'
    im = f if f.exists(
        ) else 'https://ultralytics.com/images/bus.jpg' if check_online(
        ) else np.ones((640, 640, 3))
    try:
        assert amp_allclose(model, im) or amp_allclose(DetectMultiBackend(
            'yolov5n.pt', device), im)
        LOGGER.info(f'{prefix}checks passed ✅')
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(
            f'{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}'
            )
        return False
