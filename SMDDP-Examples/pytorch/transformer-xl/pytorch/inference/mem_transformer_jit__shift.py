def _shift(self, x, qlen, klen, mask, left=False):
    if qlen > 1:
        zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
            device=x.device, dtype=x.dtype)
    else:
        zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)
    if left:
        mask = mask.flip(1)
        x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
    else:
        x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)
    x = x_padded.masked_select(mask[:, :, None, None]).view(qlen, klen, x.
        size(2), x.size(3))
    return x
