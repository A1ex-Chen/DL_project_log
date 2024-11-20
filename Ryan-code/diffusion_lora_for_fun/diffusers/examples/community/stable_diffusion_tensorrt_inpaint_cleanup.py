def cleanup(self, return_onnx=False):
    self.graph.cleanup().toposort()
    if return_onnx:
        return gs.export_onnx(self.graph)
