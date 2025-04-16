def infer_shapes(self, return_onnx=False):
    onnx_graph = gs.export_onnx(self.graph)
    if onnx_graph.ByteSize() > 2147483648:
        raise TypeError('ERROR: model size exceeds supported 2GB limit')
    else:
        onnx_graph = shape_inference.infer_shapes(onnx_graph)
    self.graph = gs.import_onnx(onnx_graph)
    if return_onnx:
        return onnx_graph
