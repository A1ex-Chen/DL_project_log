def train(self, data, labels, batch_size=100, epochs=50, validation_data=None):
    if validation_data:
        validation_size = len(validation_data[0])
    else:
        validation_size = len(data)
    print('training network on %i documents, validation on %i documents' %
        (len(data), validation_size))
    history = History()
    for ep in range(epochs):
        labels.append(data)
        xy = list(zip(*labels))
        random.shuffle(xy)
        shuffled = list(zip(*xy))
        data = list(shuffled[-1])
        labels = list(shuffled[:self.num_tasks])
        y_preds = [[] for i in range(self.num_tasks)]
        y_trues = [[] for i in range(self.num_tasks)]
        start_time = time.time()
        for start in range(0, len(data), batch_size):
            if start + batch_size < len(data):
                stop = start + batch_size
            else:
                stop = len(data)
            feed_dict = {self.doc_input: data[start:stop], self.dropout:
                self.dropout_keep}
            for i in range(self.num_tasks):
                feed_dict[self.labels[i]] = labels[i][start:stop]
            retvals = self.sess.run(self.predictions + [self.optimizer,
                self.loss], feed_dict=feed_dict)
            loss = retvals[-1]
            for i in range(self.num_tasks):
                y_preds[i].extend(np.argmax(retvals[i], 1))
                y_trues[i].extend(labels[i][start:stop])
            sys.stdout.write(
                'epoch %i, sample %i of %i, loss: %f        \r' % (ep + 1,
                stop, len(data), loss))
            sys.stdout.flush()
        print('\ntraining time: %.2f' % (time.time() - start_time))
        for i in range(self.num_tasks):
            micro = f1_score(y_trues[i], y_preds[i], average='micro')
            macro = f1_score(y_trues[i], y_preds[i], average='macro')
            print('epoch %i task %i training micro/macro: %.4f, %.4f' % (ep +
                1, i + 1, micro, macro))
        scores, val_loss = self.score(validation_data[0], validation_data[1
            ], batch_size=batch_size)
        for i in range(self.num_tasks):
            print('epoch %i task %i validation micro/macro: %.4f, %.4f' % (
                ep + 1, i + 1, scores[i][0], scores[i][1]))
        history.history.setdefault('val_loss', []).append(val_loss)
        start_time = time.time()
    return history
