def _convert_boxes(self, boxes):
    """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
        return boxes.tensor.detach().numpy()
    else:
        return np.asarray(boxes)
