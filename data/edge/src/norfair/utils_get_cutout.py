def get_cutout(points, image):
    """Returns a rectangular cut-out from a set of points on an image"""
    max_x = int(max(points[:, 0]))
    min_x = int(min(points[:, 0]))
    max_y = int(max(points[:, 1]))
    min_y = int(min(points[:, 1]))
    return image[min_y:max_y, min_x:max_x]
