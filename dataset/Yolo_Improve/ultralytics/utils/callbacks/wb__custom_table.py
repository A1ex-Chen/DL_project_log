def _custom_table(x, y, classes, title='Precision Recall Curve', x_title=
    'Recall', y_title='Precision'):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import pandas
    df = pandas.DataFrame({'class': classes, 'y': y, 'x': x}).round(3)
    fields = {'x': 'x', 'y': 'y', 'class': 'class'}
    string_fields = {'title': title, 'x-axis-title': x_title,
        'y-axis-title': y_title}
    return wb.plot_table('wandb/area-under-curve/v0', wb.Table(dataframe=df
        ), fields=fields, string_fields=string_fields)
