def evaluate_keras_metric(y_true, y_pred, metric):
    objective_function = metrics.get(metric)
    objective = objective_function(y_true, y_pred)
    return K.eval(objective)
