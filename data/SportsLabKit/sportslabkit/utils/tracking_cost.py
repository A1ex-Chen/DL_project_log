def cost(X, Y, func):
    feature = func.feature
    x_feature = getattr(X, '_feature_' + feature)
    y_feature = getattr(Y, '_feature_' + feature)
    return func(x_feature, y_feature)
