def convert(self, model: Model, dataloader_fn) ->Model:
    from .onnx import _infer_graph_precision
    from .onnx2trt_conv import onnx2trt
    pyt2onnx_converter = PYT2ONNXConverter(self._onnx_opset)
    onnx_model = pyt2onnx_converter.convert(model, dataloader_fn).handle
    precision = _infer_graph_precision(onnx_model.graph)
    input_shapes = get_input_shapes(dataloader_fn(), self._max_batch_size)
    cuda_engine = onnx2trt(onnx_model, shapes=input_shapes,
        max_workspace_size=self._max_workspace_size, max_batch_size=self.
        _max_batch_size, model_precision=self._precision.value)
    return Model(handle=cuda_engine, precision=model.precision, inputs=
        model.inputs, outputs=model.outputs)
