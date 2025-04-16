def plot_metrics(self):
    """Plots training/val metrics."""
    plot_results(file=self.csv, segment=True, on_plot=self.on_plot)
