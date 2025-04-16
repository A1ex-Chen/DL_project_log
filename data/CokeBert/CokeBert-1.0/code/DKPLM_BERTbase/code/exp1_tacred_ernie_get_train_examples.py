def get_train_examples(self, data_dir):
    """See base class."""
    examples = self._create_examples(self._read_json(os.path.join(data_dir,
        'train.json')), 'train')
    labels = set([x.label for x in examples])
    return examples, list(labels)
