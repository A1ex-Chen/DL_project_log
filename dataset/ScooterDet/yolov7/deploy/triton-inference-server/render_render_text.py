def render_text(img, text, pos, color=(200, 200, 200), normalised_scaling=1.0):
    """
    Render a text into the image. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param text: text to display
    :param pos: (x, y) - upper left coordinates of render position
    :param color: (b, g, r) - text color
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: updated image
    """
    x, y = pos
    thickness = int(round(img.shape[0] * img.shape[1] / (
        _TEXT_THICKNESS_SCALING * _TEXT_THICKNESS_SCALING)) *
        normalised_scaling)
    thickness = max(1, thickness)
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    size = get_text_size(img, text, normalised_scaling)
    cv2.putText(img, text, (int(x), int(y + size[1])), cv2.
        FONT_HERSHEY_SIMPLEX, scaling, color, thickness=thickness)
    return img
