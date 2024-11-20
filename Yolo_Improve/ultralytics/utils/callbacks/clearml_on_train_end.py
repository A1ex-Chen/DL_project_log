def on_train_end(trainer):
    """Logs final model and its name on training completion."""
    if (task := Task.current_task()):
        files = ['results.png', 'confusion_matrix.png',
            'confusion_matrix_normalized.png', *(f'{x}_curve.png' for x in
            ('F1', 'PR', 'P', 'R'))]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir /
            f).exists()]
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)
        task.update_output_model(model_path=str(trainer.best), model_name=
            trainer.args.name, auto_delete_file=False)
