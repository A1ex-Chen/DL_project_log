def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes
