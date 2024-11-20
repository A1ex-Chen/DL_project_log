def warmup(self, imgsz=(1, 3, 640, 640)):
    warmup_types = (self.pt, self.jit, self.onnx, self.engine, self.
        saved_model, self.pb)
    if any(warmup_types) and self.device.type != 'cpu':
        im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.
            float, device=self.device)
        for _ in range(2 if self.jit else 1):
            self.forward(im)
