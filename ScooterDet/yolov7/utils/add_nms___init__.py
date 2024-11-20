def __init__(self, onnx_model_path: str, precision: str='fp32'):
    self.graph = gs.import_onnx(onnx.load(onnx_model_path))
    assert self.graph
    LOGGER.info('ONNX graph created successfully')
    self.graph.fold_constants()
    self.precision = precision
    self.batch_size = 1
