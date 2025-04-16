def _getIntersectionArea(boxA: list[int], boxB: list[int]) ->int:
    """Return intersection area of two boxes.

    Args:
        boxA (list[int]): box of object
        boxB (list[int]): box of object

    Returns:
        intersection_area (int): area of intersection
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection_area = (xB - xA) * (yB - yA)
    return intersection_area
