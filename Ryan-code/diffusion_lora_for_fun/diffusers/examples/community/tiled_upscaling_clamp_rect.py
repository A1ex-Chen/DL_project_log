def clamp_rect(rect: [int], min: [int], max: [int]):
    return clamp(rect[0], min[0], max[0]), clamp(rect[1], min[1], max[1]
        ), clamp(rect[2], min[0], max[0]), clamp(rect[3], min[1], max[1])
