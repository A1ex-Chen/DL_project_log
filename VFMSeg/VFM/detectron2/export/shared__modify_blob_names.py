def _modify_blob_names(ops, blob_rename_f):
    ret = []

    def _replace_list(blob_list, replaced_list):
        del blob_list[:]
        blob_list.extend(replaced_list)
    for x in ops:
        cur = copy.deepcopy(x)
        _replace_list(cur.input, list(map(blob_rename_f, cur.input)))
        _replace_list(cur.output, list(map(blob_rename_f, cur.output)))
        ret.append(cur)
    return ret
