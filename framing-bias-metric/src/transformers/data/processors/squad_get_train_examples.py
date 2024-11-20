def get_train_examples(self, data_dir, filename=None):
    """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
    if data_dir is None:
        data_dir = ''
    if self.train_file is None:
        raise ValueError(
            'SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor'
            )
    with open(os.path.join(data_dir, self.train_file if filename is None else
        filename), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    return self._create_examples(input_data, 'train')
