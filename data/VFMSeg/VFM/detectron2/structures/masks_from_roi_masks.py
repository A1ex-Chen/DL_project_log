@staticmethod
def from_roi_masks(roi_masks: 'ROIMasks', height: int, width: int
    ) ->'BitMasks':
    """
        Args:
            roi_masks:
            height, width (int):
        """
    return roi_masks.to_bitmasks(height, width)
