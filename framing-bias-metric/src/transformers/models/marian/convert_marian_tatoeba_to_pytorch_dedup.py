def dedup(lst):
    """Preservers order"""
    new_lst = []
    for item in lst:
        if not item:
            continue
        elif item in new_lst:
            continue
        else:
            new_lst.append(item)
    return new_lst
