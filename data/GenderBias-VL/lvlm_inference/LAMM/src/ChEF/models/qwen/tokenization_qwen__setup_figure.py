def _setup_figure(self, img):
    fig = mplfigure.Figure(frameon=False)
    self.dpi = fig.get_dpi()
    fig.set_size_inches((self.width * self.scale + 0.01) / self.dpi, (self.
        height * self.scale + 0.01) / self.dpi)
    self.canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis('off')
    self.fig = fig
    self.ax = ax
    self.reset_image(img)
