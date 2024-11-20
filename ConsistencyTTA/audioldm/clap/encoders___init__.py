def __init__(self, pretrained_path='', key='class', sampling_rate=16000,
    embed_mode='audio', amodel='HTSAT-tiny', unconditional_prob=0.1,
    random_mute=False, max_random_mute_portion=0.5, training_mode=True):
    super().__init__()
    self.key = key
    self.device = 'cpu'
    self.precision = 'fp32'
    self.amodel = amodel
    self.tmodel = 'roberta'
    self.enable_fusion = False
    self.fusion_type = 'aff_2d'
    self.pretrained = pretrained_path
    self.embed_mode = embed_mode
    self.embed_mode_orig = embed_mode
    self.sampling_rate = sampling_rate
    self.unconditional_prob = unconditional_prob
    self.random_mute = random_mute
    self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
    self.max_random_mute_portion = max_random_mute_portion
    self.training_mode = training_mode
    self.model, self.model_cfg = create_model(self.amodel, self.tmodel,
        self.pretrained, precision=self.precision, device=self.device,
        enable_fusion=self.enable_fusion, fusion_type=self.fusion_type)
    for p in self.model.parameters():
        p.requires_grad = False
    self.model.eval()
