def update_pie(self, classes_dict):
    """
        Update the pie chart with new data.

        Args:
            classes_dict (dict): Dictionary containing the class data to plot.
        """
    labels = list(classes_dict.keys())
    sizes = list(classes_dict.values())
    total = sum(sizes)
    percentages = [(size / total * 100) for size in sizes]
    start_angle = 90
    self.ax.clear()
    wedges, autotexts = self.ax.pie(sizes, autopct=None, startangle=
        start_angle, textprops={'color': self.fg_color})
    legend_labels = [f'{label} ({percentage:.1f}%)' for label, percentage in
        zip(labels, percentages)]
    self.ax.legend(wedges, legend_labels, title='Classes', loc=
        'center left', bbox_to_anchor=(1, 0, 0.5, 1))
    self.fig.tight_layout()
    self.fig.subplots_adjust(left=0.1, right=0.75)
    im0 = self.fig.canvas.draw()
    im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
    self.write_and_display(im0)
