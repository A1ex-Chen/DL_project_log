def _replace_list(blob_list, replaced_list):
    del blob_list[:]
    blob_list.extend(replaced_list)
