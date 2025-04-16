def _list_find(input_list: List[Any], candidates: Tuple[Any], start: int=0):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1
