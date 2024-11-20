def decode_outputs(self, outputs, dtype):
    grids, strides = self.make_grid(dtype)
    xy = (outputs[..., :2] + grids) * strides
    wh = torch.exp(outputs[..., 2:4]) * strides
    score = outputs[..., 4:]
    out = torch.cat([xy, wh, score], -1)
    return out
