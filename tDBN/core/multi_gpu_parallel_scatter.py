def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
