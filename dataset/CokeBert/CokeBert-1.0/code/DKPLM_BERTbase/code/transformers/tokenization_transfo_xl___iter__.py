def __iter__(self):
    if self.shuffle:
        np.random.shuffle(self.paths)
    for path in self.paths:
        sent_stream = self.get_sent_stream(path)
        for batch in self.stream_iterator(sent_stream):
            yield batch
