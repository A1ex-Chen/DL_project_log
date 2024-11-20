def filter_keys(key_set):

    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}
    return _f
