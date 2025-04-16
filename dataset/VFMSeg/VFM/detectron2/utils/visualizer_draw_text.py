def draw_text(self, text, position, *, font_size=None, color='g',
    horizontal_alignment='center', rotation=0):
    """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
    if not font_size:
        font_size = self._default_font_size
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    x, y = position
    self.output.ax.text(x, y, text, size=font_size * self.output.scale,
        family='sans-serif', bbox={'facecolor': 'black', 'alpha': 0.8,
        'pad': 0.7, 'edgecolor': 'none'}, verticalalignment='top',
        horizontalalignment=horizontal_alignment, color=color, zorder=10,
        rotation=rotation)
    return self.output
