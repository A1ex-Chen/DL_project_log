def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=['best'])
    for curve_name, curve_values in zip(trainer.validator.metrics.curves,
        trainer.validator.metrics.curves_results):
        x, y, x_title, y_title = curve_values
        _plot_curve(x, y, names=list(trainer.validator.metrics.names.values
            ()), id=f'curves/{curve_name}', title=curve_name, x_title=
            x_title, y_title=y_title)
    wb.run.finish()
