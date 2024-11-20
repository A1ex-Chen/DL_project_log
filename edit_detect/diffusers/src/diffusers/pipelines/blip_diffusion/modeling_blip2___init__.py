def __init__(self, config: Blip2Config):
    super().__init__(config)
    self.config = config
    self.embeddings = Blip2TextEmbeddings(config.qformer_config)
    self.visual_encoder = Blip2VisionModel(config.vision_config)
    self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens,
        config.qformer_config.hidden_size))
    if not hasattr(config, 'tokenizer') or config.tokenizer is None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
            truncation_side='right')
    else:
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer,
            truncation_side='right')
    self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    self.proj_layer = ProjLayer(in_dim=config.qformer_config.hidden_size,
        out_dim=config.qformer_config.hidden_size, hidden_dim=config.
        qformer_config.hidden_size * 4, drop_p=0.1, eps=1e-12)
    self.encoder = Blip2QFormerEncoder(config.qformer_config)
    self.post_init()
