def _rename_blob(name, blob_sizes, blob_ranges):

    def _list_to_str(bsize):
        ret = ', '.join([str(x) for x in bsize])
        ret = '[' + ret + ']'
        return ret
    ret = name
    if blob_sizes is not None and name in blob_sizes:
        ret += '\n' + _list_to_str(blob_sizes[name])
    if blob_ranges is not None and name in blob_ranges:
        ret += '\n' + _list_to_str(blob_ranges[name])
    return ret
