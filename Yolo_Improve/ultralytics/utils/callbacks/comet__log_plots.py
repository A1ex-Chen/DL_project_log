def _log_plots(experiment, trainer):
    """Logs evaluation plots and label plots for the experiment."""
    plot_filenames = [(trainer.save_dir / f'{plots}.png') for plots in
        EVALUATION_PLOT_NAMES]
    _log_images(experiment, plot_filenames, None)
    label_plot_filenames = [(trainer.save_dir / f'{labels}.jpg') for labels in
        LABEL_PLOT_NAMES]
    _log_images(experiment, label_plot_filenames, None)
