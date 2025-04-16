def corr(y_true, y_pred):
    cov = candle.covariance(y_true, y_pred)
    var1 = candle.covariance(y_true, y_true)
    var2 = candle.covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())
