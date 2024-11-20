def plt_color_scatter(v, f, bins=20, cmap='viridis', alpha=0.8, edgecolors=
    'none'):
    """
    Plots a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [hist[min(np.digitize(v[i], xedges, right=True) - 1, hist.
        shape[0] - 1), min(np.digitize(f[i], yedges, right=True) - 1, hist.
        shape[1] - 1)] for i in range(len(v))]
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)
