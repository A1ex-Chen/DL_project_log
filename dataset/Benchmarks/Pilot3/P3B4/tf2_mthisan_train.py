def train(self, data, labels, batch_size=128, epochs=100, patience=5,
    validation_data=None, savebest=False, filepath=None):
    if savebest is True and filepath is None:
        raise Exception('Please enter a path to save the network')
    if validation_data:
        validation_size = len(validation_data[0])
    else:
        validation_size = len(data)
    print('training network on %i documents, validation on %i documents' %
        (len(data), validation_size))
    history = History()
    bestloss = np.inf
    pat_count = 0
    for ep in range(epochs):
        self.model.training = True
        labels.append(data)
        xy = list(zip(*labels))
        random.shuffle(xy)
        shuffled = list(zip(*xy))
        data = np.array(shuffled[-1]).astype(np.int32)
        labels = list(shuffled[:self.num_tasks])
        y_preds = [[] for c in self.num_classes]
        y_trues = [[] for c in self.num_classes]
        start_time = time.time()
        for start in range(0, len(data), batch_size):
            if start + batch_size < len(data):
                stop = start + batch_size
            else:
                stop = len(data)
            predictions, loss = self._train_step(data[start:stop], np.array
                ([lIndex[start:stop] for lIndex in labels]))
            for i, (p, lIndex) in enumerate(zip(predictions, [lIndex[start:
                stop] for lIndex in labels])):
                y_preds[i].extend(np.argmax(p, 1))
                y_trues[i].extend(lIndex)
            sys.stdout.write(
                'epoch %i, sample %i of %i, loss: %f        \r' % (ep + 1,
                stop, len(data), loss))
            sys.stdout.flush()
        print('\ntraining time: %.2f' % (time.time() - start_time))
        for i in range(self.num_tasks):
            micro = f1_score(y_trues[i], y_preds[i], average='micro')
            macro = f1_score(y_trues[i], y_preds[i], average='macro')
            print('epoch %i task %i training micro/macro: %.4f, %.4f' % (ep +
                1, i, micro, macro))
        scores, loss = self.score(validation_data[0], validation_data[1],
            batch_size=batch_size)
        for i in range(self.num_tasks):
            print('epoch %i task %i validation micro/macro: %.4f, %.4f' % (
                ep + 1, i, scores[i][0], scores[i][1]))
        history.history.setdefault('val_loss', []).append(loss)
        if loss < bestloss:
            bestloss = loss
            pat_count = 0
            if savebest:
                self.save(filepath)
        else:
            pat_count += 1
            if pat_count >= patience:
                break
        start_time = time.time()
    return history
