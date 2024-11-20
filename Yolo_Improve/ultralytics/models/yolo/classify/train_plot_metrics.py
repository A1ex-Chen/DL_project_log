def plot_metrics(self):
    """Plots metrics from a CSV file."""
    plot_results(file=self.csv, classify=True, on_plot=self.on_plot)
