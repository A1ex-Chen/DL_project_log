def fold_constants(self, return_onnx=False):
    onnx_graph = fold_constants(gs.export_onnx(self.graph),
        allow_onnxruntime_shape_inference=True)
    self.graph = gs.import_onnx(onnx_graph)
    if return_onnx:
        return onnx_graph
