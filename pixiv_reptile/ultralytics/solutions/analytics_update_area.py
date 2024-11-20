def update_area(self, frame_number, counts_dict):
    """
        Update the area graph with new data for multiple classes.

        Args:
            frame_number (int): The current frame number.
            counts_dict (dict): Dictionary with class names as keys and counts as values.
        """
    x_data = np.array([])
    y_data_dict = {key: np.array([]) for key in counts_dict.keys()}
    if self.ax.lines:
        x_data = self.ax.lines[0].get_xdata()
        for line, key in zip(self.ax.lines, counts_dict.keys()):
            y_data_dict[key] = line.get_ydata()
    x_data = np.append(x_data, float(frame_number))
    max_length = len(x_data)
    for key in counts_dict.keys():
        y_data_dict[key] = np.append(y_data_dict[key], float(counts_dict[key]))
        if len(y_data_dict[key]) < max_length:
            y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length -
                len(y_data_dict[key])), 'constant')
    if len(x_data) > self.max_points:
        x_data = x_data[1:]
        for key in counts_dict.keys():
            y_data_dict[key] = y_data_dict[key][1:]
    self.ax.clear()
    colors = ['#E1FF25', '#0BDBEB', '#FF64DA', '#111F68', '#042AFF']
    color_cycle = cycle(colors)
    for key, y_data in y_data_dict.items():
        color = next(color_cycle)
        self.ax.fill_between(x_data, y_data, color=color, alpha=0.6)
        self.ax.plot(x_data, y_data, color=color, linewidth=self.line_width,
            marker='o', markersize=self.points_width, label=
            f'{key} Data Points')
    self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
    self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.
        fontsize - 3)
    self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.
        fontsize - 3)
    legend = self.ax.legend(loc='upper left', fontsize=13, facecolor=self.
        bg_color, edgecolor=self.fg_color)
    for text in legend.get_texts():
        text.set_color(self.fg_color)
    self.canvas.draw()
    im0 = np.array(self.canvas.renderer.buffer_rgba())
    self.write_and_display(im0)
