def __init__(self, config, question_encoder_tokenizer, generator_tokenizer,
    index=None):
    requires_datasets(self)
    requires_faiss(self)
    super().__init__()
    self.index = index or self._build_index(config)
    self.generator_tokenizer = generator_tokenizer
    self.question_encoder_tokenizer = question_encoder_tokenizer
    self.n_docs = config.n_docs
    self.batch_size = config.retrieval_batch_size
    self.config = config
    if self._init_retrieval:
        self.init_retrieval()
