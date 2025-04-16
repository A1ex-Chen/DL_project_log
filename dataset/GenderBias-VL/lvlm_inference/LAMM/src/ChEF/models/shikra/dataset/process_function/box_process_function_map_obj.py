def map_obj(boxes_value: List[List[float]], boxes_seq: List[List[int]]) ->List[
    List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f'error: map obj {boxes_value} {boxes_seq}')
