def merge_dict_of_lists(d1: dict, d2: dict) ->dict:
    """Merge two dicts of lists.

    Parameters
    ----------
    d1 : dict
        The first dict to merge.
    d2 : dict
        The second dict to merge.

    Returns
    -------
    dict
        The merged dict.
    """
    keys = set(d1.keys()).union(d2.keys())
    ret = {k: (list(d1.get(k, [])) + list(d2.get(k, []))) for k in keys}
    return ret
