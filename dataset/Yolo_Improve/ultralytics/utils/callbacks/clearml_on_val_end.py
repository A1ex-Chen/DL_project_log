def on_val_end(validator):
    """Logs validation results including labels and predictions."""
    if Task.current_task():
        _log_debug_samples(sorted(validator.save_dir.glob('val*.jpg')),
            'Validation')
