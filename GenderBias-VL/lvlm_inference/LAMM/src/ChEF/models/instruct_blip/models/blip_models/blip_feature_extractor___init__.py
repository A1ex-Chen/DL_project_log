def __init__(self, image_encoder, text_encoder, embed_dim, max_txt_len=40):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.visual_encoder = image_encoder
    self.text_encoder = text_encoder
    text_width = text_encoder.config.hidden_size
    vision_width = image_encoder.vision_width
    self.vision_proj = nn.Linear(vision_width, embed_dim)
    self.text_proj = nn.Linear(text_width, embed_dim)
    self.max_txt_len = max_txt_len
    self.temp = nn.Parameter(0.07 * torch.ones([]))
