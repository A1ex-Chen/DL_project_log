def __init__(self, model, im):
    super().__init__()
    b, c, h, w = im.shape
    self.model = model
    self.nc = model.nc
    if w == h:
        self.normalize = 1.0 / w
    else:
        self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])
