def update_multiple_lines(self, counts_dict, labels_list, frame_number):
    """
        Update the line graph with multiple classes.

        Args:
            counts_dict (int): Dictionary include each class counts.
            labels_list (int): list include each classes names.
            frame_number (int): The current frame number.
        """
    warnings.warn(
        'Display is not supported for multiple lines, output will be stored normally!'
        )
    for obj in labels_list:
        if obj not in self.lines:
            line, = self.ax.plot([], [], label=obj, marker='o', markersize=
                self.points_width)
            self.lines[obj] = line
        x_data = self.lines[obj].get_xdata()
        y_data = self.lines[obj].get_ydata()
        if len(x_data) >= self.max_points:
            x_data = np.delete(x_data, 0)
            y_data = np.delete(y_data, 0)
        x_data = np.append(x_data, float(frame_number))
        y_data = np.append(y_data, float(counts_dict.get(obj, 0)))
        self.lines[obj].set_data(x_data, y_data)
    self.ax.relim()
    self.ax.autoscale_view()
    self.ax.legend()
    self.canvas.draw()
    im0 = np.array(self.canvas.renderer.buffer_rgba())
    self.view_img = False
    self.write_and_display(im0)
