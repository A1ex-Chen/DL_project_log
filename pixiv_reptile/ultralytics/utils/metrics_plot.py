@TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
@plt_settings()
def plot(self, normalize=True, save_dir='', names=(), on_plot=None):
    """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
    import seaborn
    array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if
        normalize else 1)
    array[array < 0.005] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = self.nc, len(names)
    seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)
    labels = 0 < nn < 99 and nn == nc
    ticklabels = list(names) + ['background'] if labels else 'auto'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seaborn.heatmap(array, ax=ax, annot=nc < 30, annot_kws={'size': 8},
            cmap='Blues', fmt='.2f' if normalize else '.0f', square=True,
            vmin=0.0, xticklabels=ticklabels, yticklabels=ticklabels
            ).set_facecolor((1, 1, 1))
    title = 'Confusion Matrix' + ' Normalized' * normalize
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(plot_fname, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(plot_fname)
