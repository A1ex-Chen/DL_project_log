def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.visual_encoder = image_encoder
    self.text_decoder = text_decoder
    self.prompt = prompt
    self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
    self.max_txt_len = max_txt_len
