def convert(self, model: Model, dataloader_fn) ->Model:
    input_shapes = get_input_shapes(dataloader_fn(), self._max_batch_size)
    cuda_engine = onnx2trt(model.handle, shapes=input_shapes,
        max_workspace_size=self._max_workspace_size, max_batch_size=self.
        _max_batch_size, model_precision=self._precision.value)
    return model._replace(handle=cuda_engine)
