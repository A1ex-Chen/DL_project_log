def setup_model(self, model, verbose=True):
    """Initialize YOLO model with given parameters and set it to evaluation mode."""
    self.model = AutoBackend(weights=model or self.args.model, device=
        select_device(self.args.device, verbose=verbose), dnn=self.args.dnn,
        data=self.args.data, fp16=self.args.half, batch=self.args.batch,
        fuse=True, verbose=verbose)
    self.device = self.model.device
    self.args.half = self.model.fp16
    self.model.eval()
