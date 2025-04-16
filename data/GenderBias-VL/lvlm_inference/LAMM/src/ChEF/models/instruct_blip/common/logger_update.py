def update(self, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        assert isinstance(v, (float, int))
        self.meters[k].update(v)
