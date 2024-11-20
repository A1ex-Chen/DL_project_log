def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    num_query_token=32, t5_model='google/flan-t5-xl', prompt='',
    max_txt_len=32, apply_lemmatizer=False):
    """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model,
        img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
    if freeze_vit:
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        logging.info('freeze vision encoder')
    self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
        self.visual_encoder.num_features)
    self.Qformer.cls = None
    self.Qformer.bert.embeddings.word_embeddings = None
    self.Qformer.bert.embeddings.position_embeddings = None
    for layer in self.Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
    t5_config = T5Config.from_pretrained(t5_model)
    t5_config.dense_act_fn = 'gelu'
    self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model,
        config=t5_config)
    for name, param in self.t5_model.named_parameters():
        param.requires_grad = False
        param.data = param.data.bfloat16()
    self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model
        .config.hidden_size)
    self.max_txt_len = max_txt_len
    self.prompt = prompt
    self._apply_lemmatizer = apply_lemmatizer
    self._lemmatizer = None
