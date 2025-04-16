@configurable
def __init__(self, tokenizer, tokenizer_type, lang_encoder, lang_projection,
    max_token_num):
    super().__init__()
    self.tokenizer = tokenizer
    self.tokenizer_type = tokenizer_type
    self.lang_encoder = lang_encoder
    self.lang_proj = lang_projection
    self.max_token_num = max_token_num
    self.logit_scale = nn.Parameter(torch.ones([]))
