def _plot_curve(x, y, names=None, id='precision-recall', title=
    'Precision Recall Curve', x_title='Recall', y_title='Precision', num_x=
    100, only_mean=False):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to [].
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()
    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title,
            y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ['mean'] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)
            y_log.extend(np.interp(x_new, x, yi))
            classes.extend([names[i]] * len(x_new))
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title,
            y_title)}, commit=False)
