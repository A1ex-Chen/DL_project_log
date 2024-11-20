def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample['disparity'].shape)
    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample
    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]
    scale = max(scale)
    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])
    sample['image'] = cv2.resize(sample['image'], tuple(shape[::-1]),
        interpolation=image_interpolation_method)
    sample['disparity'] = cv2.resize(sample['disparity'], tuple(shape[::-1]
        ), interpolation=cv2.INTER_NEAREST)
    sample['mask'] = cv2.resize(sample['mask'].astype(np.float32), tuple(
        shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample['mask'] = sample['mask'].astype(bool)
    return tuple(shape)
