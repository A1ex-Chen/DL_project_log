def __init__(self, scorer_type, model_fpath):
    """ """
    self.logger = CreateLogger()
    self.logger.debug('[model]: DensePhrasesScorer')
    self.logger.debug('[scorer_type]: %s', scorer_type)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.info('cpu/gpu: %s', self.device)
    self.nlp = spacy.load('en_core_web_sm')
    self.model_name = scorer_type if not model_fpath else model_fpath
    self.densephrases = DensePhrases(load_dir=self.model_name, dump_dir='')
    self.tokenizer = self.densephrases.tokenizer
    self.model = self.densephrases.model
    for param in self.model.parameters():
        param.requires_grad = False
    self.model.to(self.device)
    super().__init__(self.tokenizer)
