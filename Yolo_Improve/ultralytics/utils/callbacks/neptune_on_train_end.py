def on_train_end(trainer):
    """Callback function called at end of training."""
    if run:
        files = ['results.png', 'confusion_matrix.png',
            'confusion_matrix_normalized.png', *(f'{x}_curve.png' for x in
            ('F1', 'PR', 'P', 'R'))]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir /
            f).exists()]
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        run[f'weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}'
            ].upload(File(str(trainer.best)))
