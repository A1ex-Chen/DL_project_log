def _getUnionAreas(boxA: list[int], boxB: list[int], interArea: (float |
    None)=None) ->float:
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)
