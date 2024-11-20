@contextmanager
def _turn_off_roi_heads(self, attrs):
    """
        Open a context where some heads in `model.roi_heads` are temporarily turned off.
        Args:
            attr (list[str]): the attribute in `model.roi_heads` which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
    roi_heads = self.model.roi_heads
    old = {}
    for attr in attrs:
        try:
            old[attr] = getattr(roi_heads, attr)
        except AttributeError:
            pass
    if len(old.keys()) == 0:
        yield
    else:
        for attr in old.keys():
            setattr(roi_heads, attr, False)
        yield
        for attr in old.keys():
            setattr(roi_heads, attr, old[attr])
