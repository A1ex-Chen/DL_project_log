def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    num_query_token=32, opt_model='facebook/opt-2.7b', prompt='',
    max_txt_len=32, apply_lemmatizer=False):
    """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
    super().__init__()
    transformers_version = version.parse(transformers.__version__)
    assert transformers_version >= version.parse('4.27'
        ), 'BLIP-2 OPT requires transformers>=4.27'
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
    self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=
        False)
    self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=
        torch.float16)
    for name, param in self.opt_model.named_parameters():
        param.requires_grad = False
    self.eos_token_id = self.opt_tokenizer('\n', add_special_tokens=False
        ).input_ids[0]
    self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.
        opt_model.config.hidden_size)
    self.max_txt_len = max_txt_len
    self.prompt = prompt
    prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors='pt')
    self.prompt_length = prompt_tokens.attention_mask.sum(1)
    self._apply_lemmatizer = apply_lemmatizer
    self._lemmatizer = None
