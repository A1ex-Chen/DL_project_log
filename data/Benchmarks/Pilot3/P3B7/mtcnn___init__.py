def __init__(self, tasks: Dict[str, int], hparams: Hparams):
    super(MTCNN, self).__init__()
    self.hparams = hparams
    self.embed = nn.Embedding(hparams.vocab_size, hparams.embed_dim)
    self.conv1 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.
        kernel1)
    self.conv2 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.
        kernel2)
    self.conv3 = Conv1dPool(hparams.embed_dim, hparams.n_filters, hparams.
        kernel3)
    self.classifier = MultitaskClassifier(self._filter_sum(), tasks)
    self._weight_init()
