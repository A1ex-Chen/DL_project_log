def on_val_end(validator):
    """Callback function called at end of each validation."""
    if run:
        _log_images({f.stem: str(f) for f in validator.save_dir.glob(
            'val*.jpg')}, 'Validation')
