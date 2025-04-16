def draw_line(self, x_data, y_data, color, linestyle='-', linewidth=None):
    """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
    if linewidth is None:
        linewidth = self._default_font_size / 3
    linewidth = max(linewidth, 1)
    self.output.ax.add_line(mpl.lines.Line2D(x_data, y_data, linewidth=
        linewidth * self.output.scale, color=color, linestyle=linestyle))
    return self.output
