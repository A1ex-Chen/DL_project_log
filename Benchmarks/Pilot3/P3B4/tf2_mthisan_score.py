def score(self, data, labels, batch_size=128):
    self.model.training = False
    y_preds = [[] for c in self.num_classes]
    losses = []
    for start in range(0, len(data), batch_size):
        if start + batch_size < len(data):
            stop = start + batch_size
        else:
            stop = len(data)
        predictions, loss = self._score_step(data[start:stop], [lIndex[
            start:stop] for lIndex in labels])
        for i, p in enumerate(predictions):
            y_preds[i].extend(np.argmax(p, 1))
        losses.append(loss)
        sys.stdout.write('processed %i of %i records        \r' % (stop,
            len(data)))
        sys.stdout.flush()
    scores = []
    for i in range(self.num_tasks):
        micro = f1_score(labels[i], y_preds[i], average='micro')
        macro = f1_score(labels[i], y_preds[i], average='macro')
        scores.append([micro, macro])
    print()
    return scores, np.mean(losses)
