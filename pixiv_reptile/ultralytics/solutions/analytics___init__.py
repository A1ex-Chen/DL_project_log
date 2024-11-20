def __init__(self, type, writer, im0_shape, title='ultralytics', x_label=
    'x', y_label='y', bg_color='white', fg_color='black', line_color=
    'yellow', line_width=2, points_width=10, fontsize=13, view_img=False,
    save_img=True, max_points=50):
    """
        Initialize the Analytics class with various chart types.

        Args:
            type (str): Type of chart to initialize ('line', 'bar', 'pie', or 'area').
            writer (object): Video writer object to save the frames.
            im0_shape (tuple): Shape of the input image (width, height).
            title (str): Title of the chart.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            bg_color (str): Background color of the chart.
            fg_color (str): Foreground (text) color of the chart.
            line_color (str): Line color for line charts.
            line_width (int): Width of the lines in line charts.
            points_width (int): Width of line points highlighter
            fontsize (int): Font size for chart text.
            view_img (bool): Whether to display the image.
            save_img (bool): Whether to save the image.
            max_points (int): Specifies when to remove the oldest points in a graph for multiple lines.
        """
    self.bg_color = bg_color
    self.fg_color = fg_color
    self.view_img = view_img
    self.save_img = save_img
    self.title = title
    self.writer = writer
    self.max_points = max_points
    self.line_color = line_color
    self.x_label = x_label
    self.y_label = y_label
    self.points_width = points_width
    self.line_width = line_width
    self.fontsize = fontsize
    figsize = im0_shape[0] / 100, im0_shape[1] / 100
    if type in {'line', 'area'}:
        self.lines = {}
        self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
        if type == 'line':
            self.line, = self.ax.plot([], [], color=self.line_color,
                linewidth=self.line_width)
    elif type in {'bar', 'pie'}:
        self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.
            bg_color)
        self.ax.set_facecolor(self.bg_color)
        color_palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (
            214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
            (127, 127, 127), (188, 189, 34), (23, 190, 207)]
        self.color_palette = [(r / 255, g / 255, b / 255, 1) for r, g, b in
            color_palette]
        self.color_cycle = cycle(self.color_palette)
        self.color_mapping = {}
        self.ax.axis('equal') if type == 'pie' else None
    self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
    self.ax.set_xlabel(x_label, color=self.fg_color, fontsize=self.fontsize - 3
        )
    self.ax.set_ylabel(y_label, color=self.fg_color, fontsize=self.fontsize - 3
        )
    self.ax.tick_params(axis='both', colors=self.fg_color)
