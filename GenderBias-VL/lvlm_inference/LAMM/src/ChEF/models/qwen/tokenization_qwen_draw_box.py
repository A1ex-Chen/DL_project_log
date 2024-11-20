def draw_box(self, box_coord, alpha=0.5, edge_color='g', line_style='-'):
    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0
    linewidth = max(self._default_font_size / 4, 1)
    self.output.ax.add_patch(mpl.patches.Rectangle((x0, y0), width, height,
        fill=False, edgecolor=edge_color, linewidth=linewidth * self.output
        .scale, alpha=alpha, linestyle=line_style))
    return self.output
