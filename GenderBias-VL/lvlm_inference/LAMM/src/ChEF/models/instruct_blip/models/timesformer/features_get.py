def get(self, key, idx=None):
    """Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tupple, return value for each module index (ignoring output indices)
        """
    if idx is None:
        return [self.info[i][key] for i in self.out_indices]
    if isinstance(idx, (tuple, list)):
        return [self.info[i][key] for i in idx]
    else:
        return self.info[idx][key]
