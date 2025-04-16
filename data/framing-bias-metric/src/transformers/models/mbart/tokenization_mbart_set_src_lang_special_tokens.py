def set_src_lang_special_tokens(self, src_lang) ->None:
    """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
    self.cur_lang_code = self.lang_code_to_id[src_lang]
    self.prefix_tokens = []
    self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
