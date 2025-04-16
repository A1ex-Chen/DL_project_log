def add_overlap_rect(rect: [int], overlap: int, image_size: [int]):
    rect = list(rect)
    rect[0] -= overlap
    rect[1] -= overlap
    rect[2] += overlap
    rect[3] += overlap
    rect = clamp_rect(rect, [0, 0], [image_size[0], image_size[1]])
    return rect
