def __init__(self, labels, sentpiece_model=None):
    """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
        """
    self.charset = labels
    self.use_sentpiece = sentpiece_model is not None
    if self.use_sentpiece:
        self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
        self.num_labels = len(self.sentpiece)
    else:
        self.num_labels = len(self.charset)
        self.label2ind = {lab: i for i, lab in enumerate(self.charset)}
