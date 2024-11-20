def show_n_params(self, return_str=True):
    tot = 0
    for p in self.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1000000.0:
            return '{:.1f}M'.format(tot / 1000000.0)
        else:
            return '{:.1f}K'.format(tot / 1000.0)
    else:
        return tot
