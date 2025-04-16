def __init__(self, scorer_type, model_fpath):
    """ """
    self.logger = CreateLogger()
    self.logger.debug('[model]: BertScorer')
    self.logger.debug('[scorer_type]: %s', scorer_type)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.info('cpu/gpu: %s', self.device)
    self.model_name = scorer_type if not model_fpath else model_fpath
    self.config = AutoConfig.from_pretrained(self.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=
        self.model_name, config=self.config)
    for param in self.model.parameters():
        param.requires_grad = False
    self.model.to(self.device)
    super().__init__(self.tokenizer)
