def _select_batch_size(self, lower, upper, is_increasing):
    diff = upper - lower
    base = lower if is_increasing else upper
    mult = 1 if is_increasing else -1
    tiers = [100, 20, 10, 5]
    for t in tiers:
        if diff >= t:
            return base + mult * t
    return base + mult
