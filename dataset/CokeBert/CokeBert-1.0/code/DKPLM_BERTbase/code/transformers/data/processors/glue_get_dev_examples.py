def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self._read_tsv(os.path.join(data_dir,
        'dev.tsv')), 'dev')
