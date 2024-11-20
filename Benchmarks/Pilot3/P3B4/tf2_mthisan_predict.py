def predict(self, data, batch_size=128):
    self.model.training = False
    y_preds = [[] for c in self.num_classes]
    for start in range(0, len(data), batch_size):
        if start + batch_size < len(data):
            stop = start + batch_size
        else:
            stop = len(data)
        predictions = self._predict_step(data[start:stop])
        for i, p in enumerate(predictions):
            y_preds[i].extend(np.argmax(p, 1))
        sys.stdout.write('processed %i of %i records        \r' % (stop,
            len(data)))
        sys.stdout.flush()
    print()
    return y_preds
