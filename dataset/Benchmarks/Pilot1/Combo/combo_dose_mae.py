def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)
