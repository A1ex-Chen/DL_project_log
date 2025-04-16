def _get_upper_left_courner(self, hull):
    """Find the nearest point from the upper left corner."""
    sorted_hull = sorted(hull, key=lambda x: x[0][0] * x[0][0] + x[0][1] *
        x[0][1])
    return sorted_hull[0][0]
