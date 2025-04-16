def _list_to_str(bsize):
    ret = ', '.join([str(x) for x in bsize])
    ret = '[' + ret + ']'
    return ret
