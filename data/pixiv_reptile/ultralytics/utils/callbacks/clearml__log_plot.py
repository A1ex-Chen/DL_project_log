def _log_plot(title, plot_path) ->None:
    """
    Log an image as a plot in the plot section of ClearML.

    Args:
        title (str): The title of the plot.
        plot_path (str): The path to the saved image file.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='auto', xticks=[],
        yticks=[])
    ax.imshow(img)
    Task.current_task().get_logger().report_matplotlib_figure(title=title,
        series='', figure=fig, report_interactive=False)
