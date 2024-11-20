def add_bbox_to_frame(image: np.ndarray, left: int, top: int, right: int,
    bottom: int, label: (str | None)=None, color: (str | None)=None
    ) ->np.ndarray:
    """Add bounding box and label to image.

    Args:
        img (np.ndarray): Image.
        left (int): Bounding box left coordinate.
        top (int): Bounding box top coordinate.
        right (int): Bounding box right coordinate.
        bottom (int): Bounding box bottom coordinate.
        label (str): Label.
        color (str): Color.
    Returns:
        img (np.ndarray): Image with bounding box and label.
    """
    _DEFAULT_COLOR_NAME = 'purple'
    if isinstance(image, np.ndarray) is False:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError as e:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number"
            ) from e
    if label and isinstance(label, str) is False:
        raise TypeError("'label' must be a str")
    if label and not color:
        hex_digest = md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]
    if not color:
        color = _DEFAULT_COLOR_NAME
    if isinstance(color, str) is False:
        raise TypeError("'color' must be a str")
    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ', '.join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)
    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color_value, _ = colors
    image = cv2.rectangle(image, (left, top), (right, bottom), color_value, 2)
    if label:
        _, image_width, _ = image.shape
        fontface = cv2.FONT_HERSHEY_TRIPLEX
        fontscale = 0.5
        thickness = 1
        (label_width, label_height), _ = cv2.getTextSize(label, fontface,
            fontscale, thickness)
        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width
        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))
        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width
        label_top = rectangle_top + 1
        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1
            label_top = rectangle_top
        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        rec_left_top = rectangle_left, rectangle_top
        rec_right_bottom = rectangle_right, rectangle_bottom
        cv2.rectangle(image, rec_left_top, rec_right_bottom, color_value, -1)
        cv2.putText(image, text=label, org=(label_left, int(label_bottom)),
            fontFace=fontface, fontScale=fontscale, color=(0, 0, 0),
            thickness=thickness, lineType=cv2.LINE_4)
    return image
