def loss(y_true, y_pred):
    cost = 0
    base_pred = (1 - mask) * y_pred
    base_true = y_true
    base_cost = K.sparse_categorical_crossentropy(base_true, base_pred)
    abs_pred = K.mean(mask * y_pred, axis=-1)
    abs_pred = K.clip(abs_pred, K.epsilon(), 1.0 - K.epsilon())
    cost = (1.0 - abs_pred) * base_cost - alpha * K.log(1.0 - abs_pred)
    return cost
