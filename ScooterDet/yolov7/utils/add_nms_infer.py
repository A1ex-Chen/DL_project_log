def infer(self):
    """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort,
        and fold constant inputs values. When possible, run shape inference on the
        ONNX graph to determine tensor shapes.
        """
    for _ in range(3):
        count_before = len(self.graph.nodes)
        self.graph.cleanup().toposort()
        try:
            for node in self.graph.nodes:
                for o in node.outputs:
                    o.shape = None
            model = gs.export_onnx(self.graph)
            model = shape_inference.infer_shapes(model)
            self.graph = gs.import_onnx(model)
        except Exception as e:
            LOGGER.info(
                f'Shape inference could not be performed at this time:\n{e}')
        try:
            self.graph.fold_constants(fold_shapes=True)
        except TypeError as e:
            LOGGER.error(
                f"""This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your onnx_graphsurgeon module. Error:
{e}"""
                )
            raise
        count_after = len(self.graph.nodes)
        if count_before == count_after:
            break
