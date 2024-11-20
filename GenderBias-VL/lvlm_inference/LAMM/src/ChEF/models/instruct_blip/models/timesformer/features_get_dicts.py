def get_dicts(self, keys=None, idx=None):
    """return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)"""
    if idx is None:
        if keys is None:
            return [self.info[i] for i in self.out_indices]
        else:
            return [{k: self.info[i][k] for k in keys} for i in self.
                out_indices]
    if isinstance(idx, (tuple, list)):
        return [(self.info[i] if keys is None else {k: self.info[i][k] for
            k in keys}) for i in idx]
    else:
        return self.info[idx] if keys is None else {k: self.info[idx][k] for
            k in keys}
