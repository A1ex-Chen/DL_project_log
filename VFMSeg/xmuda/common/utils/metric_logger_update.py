def update(self, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            count = v.numel()
            value = v.item() if count == 1 else v.sum().item()
        elif isinstance(v, np.ndarray):
            count = v.size
            value = v.item() if count == 1 else v.sum().item()
        else:
            assert isinstance(v, (float, int))
            value = v
            count = 1
        self.meters[k].update(value, count)
