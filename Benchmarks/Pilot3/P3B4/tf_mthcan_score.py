def score(self, data, labels, batch_size=16):
    loss = []
    y_preds = [[] for i in range(self.num_tasks)]
    for start in range(0, len(data), batch_size):
        if start + batch_size < len(data):
            stop = start + batch_size
        else:
            stop = len(data)
        feed_dict = {self.doc_input: data[start:stop], self.dropout: 1.0}
        for i in range(self.num_tasks):
            feed_dict[self.labels[i]] = labels[i][start:stop]
        retvals = self.sess.run(self.predictions + [self.loss], feed_dict=
            feed_dict)
        loss.append(retvals[-1])
        for i in range(self.num_tasks):
            y_preds[i].append(np.argmax(retvals[i], 1))
        sys.stdout.write('processed %i of %i records        \r' % (stop,
            len(data)))
        sys.stdout.flush()
    loss = np.mean(loss)
    print()
    for i in range(self.num_tasks):
        y_preds[i] = np.concatenate(y_preds[i], 0)
    scores = []
    for i in range(self.num_tasks):
        micro = f1_score(labels[i], y_preds[i], average='micro')
        macro = f1_score(labels[i], y_preds[i], average='macro')
        scores.append((micro, macro))
    return scores, loss
