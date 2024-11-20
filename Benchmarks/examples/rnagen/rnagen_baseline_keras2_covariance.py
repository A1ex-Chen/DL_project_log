def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)
