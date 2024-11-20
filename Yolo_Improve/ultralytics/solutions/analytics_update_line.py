def update_line(self, frame_number, total_counts):
    """
        Update the line graph with new data.

        Args:
            frame_number (int): The current frame number.
            total_counts (int): The total counts to plot.
        """
    x_data = self.line.get_xdata()
    y_data = self.line.get_ydata()
    x_data = np.append(x_data, float(frame_number))
    y_data = np.append(y_data, float(total_counts))
    self.line.set_data(x_data, y_data)
    self.ax.relim()
    self.ax.autoscale_view()
    self.canvas.draw()
    im0 = np.array(self.canvas.renderer.buffer_rgba())
    self.write_and_display(im0)
