def get_train_examples(self, data_dir):
    """See base class."""
    logger.info('LOOKING AT {}'.format(os.path.join(data_dir, 'train.json')))
    examples = self._create_examples(self._read_json(os.path.join(data_dir,
        'train.json')), 'train')
    d = {}
    for e in examples:
        for l in e.label:
            if l in d:
                d[l] += 1
            else:
                d[l] = 1
    for k, v in d.items():
        d[k] = (len(examples) - v) * 1.0 / v
    return examples, list(d.keys()), d
