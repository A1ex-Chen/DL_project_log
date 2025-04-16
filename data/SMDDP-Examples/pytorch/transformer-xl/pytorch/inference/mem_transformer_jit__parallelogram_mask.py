def _parallelogram_mask(self, h, w, left=False):
    mask = torch.ones((h, w)).byte()
    m = min(h, w)
    mask[:m, :m] = torch.triu(mask[:m, :m])
    mask[-m:, -m:] = torch.tril(mask[-m:, -m:])
    if left:
        return mask.bool()
    else:
        return mask.flip(0).bool()
