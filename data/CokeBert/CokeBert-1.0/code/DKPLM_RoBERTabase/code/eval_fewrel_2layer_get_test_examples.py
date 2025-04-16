def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self._read_json(os.path.join(data_dir,
        'test.json')), 'dev')
