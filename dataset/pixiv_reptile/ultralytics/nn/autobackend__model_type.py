@staticmethod
def _model_type(p='path/to/model.pt'):
    """
        This function takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml,
        engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

        Args:
            p: path to the model file. Defaults to path/to/model.pt

        Examples:
            >>> model = AutoBackend(weights="path/to/model.onnx")
            >>> model_type = model._model_type()  # returns "onnx"
        """
    from ultralytics.engine.exporter import export_formats
    sf = list(export_formats().Suffix)
    if not is_url(p) and not isinstance(p, str):
        check_suffix(p, sf)
    name = Path(p).name
    types = [(s in name) for s in sf]
    types[5] |= name.endswith('.mlmodel')
    types[8] &= not types[9]
    if any(types):
        triton = False
    else:
        from urllib.parse import urlsplit
        url = urlsplit(p)
        triton = bool(url.netloc) and bool(url.path) and url.scheme in {'http',
            'grpc'}
    return types + [triton]
