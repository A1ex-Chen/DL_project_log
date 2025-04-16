def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=35):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.text_encoder = text_encoder
    self.visual_encoder = image_encoder
    self.max_txt_len = max_txt_len
    text_width = text_encoder.config.hidden_size
    vision_width = image_encoder.vision_width
    self.vision_proj = nn.Linear(vision_width, embed_dim)
    self.text_proj = nn.Linear(text_width, embed_dim)
    self.itm_head = nn.Linear(text_width, 2)
