def forward(self, x):
    downcast_x = _cast_if_autocast_enabled(x)
    downcast_weight = _cast_if_autocast_enabled(self.weight
        ) if self.weight is not None else self.weight
    with torch.autocast(enabled=False, device_type=x.device.type):
        return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype
            )
