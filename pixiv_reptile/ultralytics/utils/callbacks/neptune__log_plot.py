def _log_plot(title, plot_path):
    """
    Log plots to the NeptuneAI experiment logger.

    Args:
        title (str): Title of the plot.
        plot_path (PosixPath | str): Path to the saved image file.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='auto', xticks=[],
        yticks=[])
    ax.imshow(img)
    run[f'Plots/{title}'].upload(fig)
