def save(self, output_path):
    """
        Save the ONNX model to the given location.
        Args:
            output_path: Path pointing to the location where to write
                out the updated ONNX model.
        """
    self.graph.cleanup().toposort()
    model = gs.export_onnx(self.graph)
    onnx.save(model, output_path)
    LOGGER.info(f'Saved ONNX model to {output_path}')
