def set_tgt_lang_special_tokens(self, lang: str) ->None:
    """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
    self.cur_lang_code = self.lang_code_to_id[lang]
    self.prefix_tokens = []
    self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
