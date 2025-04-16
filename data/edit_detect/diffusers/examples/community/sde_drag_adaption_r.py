def adaption_r(source, target, adapt_radius, max_height, max_width):
    r_x_lower = min(adapt_radius, source[0], target[0])
    r_x_upper = min(adapt_radius, max_width - source[0], max_width - target[0])
    r_y_lower = min(adapt_radius, source[1], target[1])
    r_y_upper = min(adapt_radius, max_height - source[1], max_height -
        target[1])
    return r_x_lower, r_x_upper, r_y_lower, r_y_upper
