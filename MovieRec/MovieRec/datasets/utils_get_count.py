def get_count(tp, id):
    groups = tp[[id]].groupby(id, as_index=False)
    count = groups.size()
    return count
