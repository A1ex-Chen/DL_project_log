def _boxesIntersect(boxA: list[int], boxB: list[int]) ->bool:
    """Checking the position of two boxes.

    Args:
        boxA (list[int]): box of object
        boxB (list[int]): box of object

    Returns:
        bool: True if boxes intersect, False otherwise
    """
    if boxA[0] > boxB[2]:
        return False
    if boxB[0] > boxA[2]:
        return False
    if boxA[3] < boxB[1]:
        return False
    if boxA[1] > boxB[3]:
        return False
    return True
