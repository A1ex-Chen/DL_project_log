def crop_len(orig_len, n_crops, overlap):
    """Crops bounding boxes to the size of the input image."""
    return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
