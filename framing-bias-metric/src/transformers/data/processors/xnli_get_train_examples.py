def get_train_examples(self, data_dir):
    """See base class."""
    lg = self.language if self.train_language is None else self.train_language
    lines = self._read_tsv(os.path.join(data_dir,
        'XNLI-MT-1.0/multinli/multinli.train.{}.tsv'.format(lg)))
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = '%s-%s' % ('train', i)
        text_a = line[0]
        text_b = line[1]
        label = 'contradiction' if line[2] == 'contradictory' else line[2]
        assert isinstance(text_a, str
            ), f'Training input {text_a} is not a string'
        assert isinstance(text_b, str
            ), f'Training input {text_b} is not a string'
        assert isinstance(label, str
            ), f'Training label {label} is not a string'
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=
            text_b, label=label))
    return examples
