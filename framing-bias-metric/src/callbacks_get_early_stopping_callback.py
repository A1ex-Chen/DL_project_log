def get_early_stopping_callback(metric, patience):
    return EarlyStopping(monitor=f'val_{metric}', mode='min' if 'loss' in
        metric else 'max', patience=patience, verbose=True)
