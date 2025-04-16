def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val
