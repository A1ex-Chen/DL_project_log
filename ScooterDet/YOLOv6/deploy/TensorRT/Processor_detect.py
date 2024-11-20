def detect(self, img):
    """Detect objects in the input image."""
    resized, _ = self.pre_process(img, self.input_shape)
    outputs = self.inference(resized)
    return outputs
