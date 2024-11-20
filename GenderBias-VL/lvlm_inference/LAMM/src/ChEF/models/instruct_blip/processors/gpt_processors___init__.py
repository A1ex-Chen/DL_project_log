def __init__(self, visual_ft, audio_ft):
    super().__init__(visual_ft, audio_ft)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
