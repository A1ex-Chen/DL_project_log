def merge_dicts(*dicts):
    """Merge dictionaries.

    Later dictionaries take precedence.
    """
    merged = {}
    for d in dicts:
        if d is not None:
            merged.update(d)
    return merged
