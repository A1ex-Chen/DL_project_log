def draw_circle(self, circle_coord, color, radius=3):
    """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
    x, y = circle_coord
    self.output.ax.add_patch(mpl.patches.Circle(circle_coord, radius=radius,
        fill=True, color=color))
    return self.output
