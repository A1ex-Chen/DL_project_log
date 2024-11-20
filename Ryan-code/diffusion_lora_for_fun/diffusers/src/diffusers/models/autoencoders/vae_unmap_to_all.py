def unmap_to_all(self, inds: torch.LongTensor) ->torch.LongTensor:
    ishape = inds.shape
    assert len(ishape) > 1
    inds = inds.reshape(ishape[0], -1)
    used = self.used.to(inds)
    if self.re_embed > self.used.shape[0]:
        inds[inds >= self.used.shape[0]] = 0
    back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
    return back.reshape(ishape)
