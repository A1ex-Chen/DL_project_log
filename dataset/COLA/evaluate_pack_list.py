def pack_list(my_list, pl=4):
    packed_list = [my_list[my_idx:my_idx + pl] for my_idx in range(0, len(
        my_list), pl)]
    if isinstance(my_list, np.ndarray):
        packed_list = np.array(packed_list)
    return packed_list
