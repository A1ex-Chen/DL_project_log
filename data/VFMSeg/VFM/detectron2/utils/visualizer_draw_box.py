def draw_box(self, box_coord, alpha=0.5, edge_color='g', line_style='-'):
    """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0
    linewidth = max(self._default_font_size / 4, 1)
    self.output.ax.add_patch(mpl.patches.Rectangle((x0, y0), width, height,
        fill=False, edgecolor=edge_color, linewidth=linewidth * self.output
        .scale, alpha=alpha, linestyle=line_style))
    return self.output
