def duplicate_list(my_list, length):
    idx, new_list = 0, []
    while len(new_list) < length:
        new_list.append(my_list[idx])
        idx = (idx + 1) % len(my_list)
    return new_list
