def _clip_augmented(self, y):
    """Clip YOLO augmented inference tails."""
    nl = self.model[-1].nl
    g = sum(4 ** x for x in range(nl))
    e = 1
    i = y[0].shape[-1] // g * sum(4 ** x for x in range(e))
    y[0] = y[0][..., :-i]
    i = y[-1].shape[-1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
    y[-1] = y[-1][..., i:]
    return y
