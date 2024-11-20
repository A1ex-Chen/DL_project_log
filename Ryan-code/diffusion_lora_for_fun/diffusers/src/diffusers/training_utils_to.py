def to(self, device=None, dtype=None) ->None:
    """Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
    self.shadow_params = [(p.to(device=device, dtype=dtype) if p.
        is_floating_point() else p.to(device=device)) for p in self.
        shadow_params]
