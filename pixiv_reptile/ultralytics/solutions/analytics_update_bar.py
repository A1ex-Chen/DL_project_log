def update_bar(self, count_dict):
    """
        Update the bar graph with new data.

        Args:
            count_dict (dict): Dictionary containing the count data to plot.
        """
    self.ax.clear()
    self.ax.set_facecolor(self.bg_color)
    labels = list(count_dict.keys())
    counts = list(count_dict.values())
    for label in labels:
        if label not in self.color_mapping:
            self.color_mapping[label] = next(self.color_cycle)
    colors = [self.color_mapping[label] for label in labels]
    bars = self.ax.bar(labels, counts, color=colors)
    for bar, count in zip(bars, counts):
        self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(count), ha='center', va='bottom', color=self.fg_color)
    canvas = FigureCanvas(self.fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    im0 = np.asarray(buf)
    self.write_and_display(im0)
