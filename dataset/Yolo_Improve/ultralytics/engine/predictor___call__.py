def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
    """Performs inference on an image or stream."""
    self.stream = stream
    if stream:
        return self.stream_inference(source, model, *args, **kwargs)
    else:
        return list(self.stream_inference(source, model, *args, **kwargs))
