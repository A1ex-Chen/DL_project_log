def name_onnx_nodes(self, model_path):
    """
        Name all unnamed nodes in ONNX model
            parameter model_path: path  ONNX model
            return: none
        """
    model = onnx.load(model_path)
    node_id = 0
    for node in model.graph.node:
        if len(node.name) == 0:
            node.name = 'unnamed_node_%d' % node_id
        node_id += 1
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    onnxruntime.InferenceSession(model_path, None)
