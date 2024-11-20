def add_frame_id_to_frame(image: np.ndarray, frame_id: int) ->np.ndarray:
    """Add frame id to image.

    Args:
        img (np.ndarray): Image.
        frame_id (int): Frame id.
    Returns:
        img (np.ndarray): Image with frame id.
    """
    if isinstance(image, np.ndarray) is False:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        frame_id = int(frame_id)
    except ValueError as e:
        raise TypeError("'frame_id' must be a number") from e
    fontface = cv2.FONT_HERSHEY_TRIPLEX
    fontscale = 5
    thickness = 1
    label = f'frame: {frame_id}'
    (label_width, label_height), _ = cv2.getTextSize(label, fontface,
        fontscale, thickness)
    label_left = 10
    label_bottom = label_height + 10
    cv2.putText(image, text=label, org=(label_left, int(label_bottom)),
        fontFace=fontface, fontScale=fontscale, color=(0, 0, 0), thickness=
        thickness, lineType=cv2.LINE_4)
    return image
