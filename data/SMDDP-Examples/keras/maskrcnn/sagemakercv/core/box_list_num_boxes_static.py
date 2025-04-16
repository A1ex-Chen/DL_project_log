def num_boxes_static(self):
    """Returns number of boxes held in collection.

    This number is inferred at graph construction time rather than run-time.

    Returns:
      Number of boxes held in collection (integer) or None if this is not
        inferrable at graph construction time.
    """
    try:
        return self.data['boxes'].get_shape()[0].value
    except AttributeError:
        return self.data['boxes'].get_shape()[0]
