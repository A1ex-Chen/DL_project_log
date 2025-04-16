def _convert_xywh_to_x1y1x2y2(bbox: list[int]) ->list[int]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]
