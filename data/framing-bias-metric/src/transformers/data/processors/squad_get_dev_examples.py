def get_dev_examples(self, data_dir, filename=None):
    """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
    if data_dir is None:
        data_dir = ''
    if self.dev_file is None:
        raise ValueError(
            'SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor'
            )
    with open(os.path.join(data_dir, self.dev_file if filename is None else
        filename), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    return self._create_examples(input_data, 'dev')
