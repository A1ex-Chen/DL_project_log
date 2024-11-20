def convert_to_x1y1x2y2(bbox: list[int]) ->list[int]:
    """Convert bbox to x1y1x2y2 format."""
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return [x1, y1, x2, y2]
