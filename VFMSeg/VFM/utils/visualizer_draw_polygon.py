def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
    """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
    if edge_color is None:
        if alpha > 0.8:
            edge_color = self._change_color_brightness(color,
                brightness_factor=-0.7)
        else:
            edge_color = color
    edge_color = mplc.to_rgb(edge_color) + (1,)
    polygon = mpl.patches.Polygon(segment, fill=True, facecolor=mplc.to_rgb
        (color) + (alpha,), edgecolor=edge_color, linewidth=max(self.
        _default_font_size // 15 * self.output.scale, 1))
    self.output.ax.add_patch(polygon)
    return self.output
