def evaluate_model(model, generator, steps, metric, category_cutoffs=[0.0]):
    y_true, y_pred = None, None
    count = 0
    while count < steps:
        x_batch, y_batch = next(generator)
        y_batch_pred = model.predict_on_batch(x_batch)
        y_batch_pred = y_batch_pred.ravel()
        y_true = np.concatenate((y_true, y_batch)
            ) if y_true is not None else y_batch
        y_pred = np.concatenate((y_pred, y_batch_pred)
            ) if y_pred is not None else y_batch_pred
        count += 1
    loss = evaluate_keras_metric(y_true.astype(np.float32), y_pred.astype(
        np.float32), metric)
    y_true_class = np.digitize(y_true, category_cutoffs)
    y_pred_class = np.digitize(y_pred, category_cutoffs)
    acc = evaluate_keras_metric(y_true_class.astype(np.float32),
        y_pred_class.astype(np.float32), 'binary_accuracy')
    return loss, acc, y_true, y_pred, y_true_class, y_pred_class
