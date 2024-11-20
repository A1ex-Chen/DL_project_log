def predict(self, data, batch_size=100):
    y_preds = [[] for i in range(self.num_tasks)]
    for start in range(0, len(data), batch_size):
        if start + batch_size < len(data):
            stop = start + batch_size
        else:
            stop = len(data)
        feed_dict = {self.doc_input: data[start:stop], self.dropout: 1.0}
        preds = self.sess.run(self.predictions, feed_dict=feed_dict)
        for i in range(self.num_tasks):
            y_preds[i].append(np.argmax(preds[i], 1))
        sys.stdout.write('processed %i of %i records        \r' % (stop,
            len(data)))
        sys.stdout.flush()
    print()
    for i in range(self.num_tasks):
        y_preds[i] = np.concatenate(y_preds[i], 0)
    return y_preds
