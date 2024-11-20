def __init__(self, im0) ->None:
    """Initialize Tensor Dataloader."""
    self.im0 = self._single_check(im0)
    self.bs = self.im0.shape[0]
    self.mode = 'image'
    self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in
        enumerate(im0)]
