def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model. If the checks
    fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will
    be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Example:
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_amp

        model = YOLO('yolov8n.pt').model.cuda()
        check_amp(model)
        ```

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.
    """
    device = next(model.parameters()).device
    if device.type in {'cpu', 'mps'}:
        return False

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        a = m(im, device=device, verbose=False)[0].boxes.data
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.data
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)
    im = ASSETS / 'bus.jpg'
    prefix = colorstr('AMP: ')
    LOGGER.info(
        f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...'
        )
    warning_msg = (
        "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
        )
    try:
        from ultralytics import YOLO
        assert amp_allclose(YOLO('yolov8n.pt'), im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except ConnectionError:
        LOGGER.warning(
            f'{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}'
            )
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f'{prefix}checks skipped ⚠️. Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}'
            )
    except AssertionError:
        LOGGER.warning(
            f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to NaN losses or zero-mAP results, so AMP will be disabled during training.'
            )
        return False
    return True
