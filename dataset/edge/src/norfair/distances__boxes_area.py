def _boxes_area(boxes: np.ndarray) ->np.ndarray:
    """
    Calculate the area of bounding boxes.
    """
    return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
