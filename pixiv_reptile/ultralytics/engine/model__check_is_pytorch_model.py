def _check_is_pytorch_model(self) ->None:
    """Raises TypeError is model is not a PyTorch model."""
    pt_str = isinstance(self.model, (str, Path)) and Path(self.model
        ).suffix == '.pt'
    pt_module = isinstance(self.model, nn.Module)
    if not (pt_module or pt_str):
        raise TypeError(
            f"""model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.
To run CUDA or MPS inference please pass the device argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"""
            )
