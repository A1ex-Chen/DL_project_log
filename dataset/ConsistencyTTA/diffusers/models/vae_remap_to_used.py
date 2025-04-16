def remap_to_used(self, inds):
    ishape = inds.shape
    assert len(ishape) > 1
    inds = inds.reshape(ishape[0], -1)
    used = self.used.to(inds)
    match = (inds[:, :, None] == used[None, None, ...]).long()
    new = match.argmax(-1)
    unknown = match.sum(2) < 1
    if self.unknown_index == 'random':
        new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape
            ).to(device=new.device)
    else:
        new[unknown] = self.unknown_index
    return new.reshape(ishape)
