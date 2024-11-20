def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=35):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.visual_encoder = image_encoder
    self.text_encoder = text_encoder
    self.text_decoder = text_decoder
    self.max_txt_len = max_txt_len
