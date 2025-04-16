def __init__(self, image_encoder, text_encoder, text_decoder, queue_size,
    alpha=0.4, embed_dim=256, momentum=0.995, tie_enc_dec_weights=True,
    max_txt_len=30):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    text_encoder.resize_token_embeddings(len(self.tokenizer))
    text_decoder.resize_token_embeddings(len(self.tokenizer))
    if tie_enc_dec_weights:
        tie_encoder_decoder_weights(encoder=text_encoder, decoder=
            text_decoder.bert, base_model_prefix='', skip_key='/attention')
    self.visual_encoder = image_encoder
    self.text_encoder = text_encoder
    self.text_decoder = text_decoder
    text_width = text_encoder.config.hidden_size
    vision_width = image_encoder.vision_width
    self.vision_proj = nn.Linear(vision_width, embed_dim)
    self.text_proj = nn.Linear(text_width, embed_dim)
    self.itm_head = nn.Linear(text_width, 2)
    self.visual_encoder_m = deepcopy(self.visual_encoder)
    self.text_encoder_m = deepcopy(self.text_encoder)
    self.vision_proj_m = deepcopy(self.vision_proj)
    self.text_proj_m = deepcopy(self.text_proj)
    self.model_pairs = [[self.visual_encoder, self.visual_encoder_m], [self
        .text_encoder, self.text_encoder_m], [self.vision_proj, self.
        vision_proj_m], [self.text_proj, self.text_proj_m]]
    self.copy_params()
    self.register_buffer('image_queue', torch.randn(embed_dim, queue_size))
    self.register_buffer('text_queue', torch.randn(embed_dim, queue_size))
    self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
    self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
    self.queue_size = queue_size
    self.momentum = momentum
    self.temp = nn.Parameter(0.07 * torch.ones([]))
    self.alpha = alpha
    self.max_txt_len = max_txt_len
