def to(self, device):
    cast_tensor = self.tensors.to(device)
    mask = self.mask
    if mask is not None:
        assert mask is not None
        cast_mask = mask.to(device)
    else:
        cast_mask = None
    return NestedTensor(cast_tensor, cast_mask)
