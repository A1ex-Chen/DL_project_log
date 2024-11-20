def draw_text(self, text, position, *, font_size=None, color='g',
    horizontal_alignment='center', rotation=0):
    if not font_size:
        font_size = self._default_font_size
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    x, y = position
    self.output.ax.text(x, y, text, size=font_size * self.output.scale,
        fontproperties=FontProperties(fname=self.font_path), bbox={
        'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'
        }, verticalalignment='top', horizontalalignment=
        horizontal_alignment, color=color, zorder=10, rotation=rotation)
    return self.output
