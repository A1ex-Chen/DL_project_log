def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
    if not (self.comet_log_predictions and (batch_i + 1) % self.
        comet_log_prediction_interval == 0):
        return
    for si, pred in enumerate(outputs):
        if len(pred) == 0:
            continue
        image = images[si]
        labels = targets[targets[:, 0] == si, 1:]
        shape = shapes[si]
        path = paths[si]
        predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)
        if labelsn is not None:
            self.log_predictions(image, labelsn, path, shape, predn)
    return
