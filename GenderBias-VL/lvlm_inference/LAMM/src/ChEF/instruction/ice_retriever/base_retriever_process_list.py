def process_list(self, near_ids, idx, ice_num):
    if idx in near_ids:
        near_ids.remove(idx)
    else:
        near_ids = near_ids[:ice_num]
    return near_ids
