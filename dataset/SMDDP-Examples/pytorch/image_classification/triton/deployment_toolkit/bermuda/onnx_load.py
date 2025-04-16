def load(self, model_path: Union[str, Path], **_) ->Model:
    if isinstance(model_path, Path):
        model_path = model_path.as_posix()
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    model = onnx.shape_inference.infer_shapes(model)
    inputs = {vi.name: _value_info2tensor_spec(vi) for vi in model.graph.input}
    outputs = {vi.name: _value_info2tensor_spec(vi) for vi in model.graph.
        output}
    precision = _infer_graph_precision(model.graph)
    return Model(model, precision, inputs, outputs)
