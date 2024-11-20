def render_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    thickness = int(round(img.shape[0] * img.shape[1] / (
        _LINE_THICKNESS_SCALING * _LINE_THICKNESS_SCALING)))
    thickness = max(1, thickness)
    img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
        thickness=thickness)
    return img
