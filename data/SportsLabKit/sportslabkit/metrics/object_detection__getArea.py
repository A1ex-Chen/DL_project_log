def _getArea(box: list[int]) ->int:
    """Return area of box.

    Args:
        box (list[int]): box of object

    Returns:
        area (int): area of box
    """
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area
