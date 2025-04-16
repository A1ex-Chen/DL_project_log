def __init__(self, model=None, **kwargs):
    logger.info(
        '`diffusers.OnnxRuntimeModel` is experimental and might change in the future.'
        )
    self.model = model
    self.model_save_dir = kwargs.get('model_save_dir', None)
    self.latest_model_name = kwargs.get('latest_model_name', ONNX_WEIGHTS_NAME)
