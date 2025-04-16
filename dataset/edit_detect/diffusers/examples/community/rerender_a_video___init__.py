def __init__(self, dims, mode='sintel', padding_factor=8):
    self.ht, self.wd = dims[-2:]
    pad_ht = ((self.ht // padding_factor + 1) * padding_factor - self.ht
        ) % padding_factor
    pad_wd = ((self.wd // padding_factor + 1) * padding_factor - self.wd
        ) % padding_factor
    if mode == 'sintel':
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht -
            pad_ht // 2]
    else:
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
