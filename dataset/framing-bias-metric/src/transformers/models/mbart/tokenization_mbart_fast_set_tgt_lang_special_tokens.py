def set_tgt_lang_special_tokens(self, lang: str) ->None:
    """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
    self.cur_lang_code = self.convert_tokens_to_ids(lang)
    self.prefix_tokens = []
    self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
    prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
    suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
    self._tokenizer.post_processor = processors.TemplateProcessing(single=
        prefix_tokens_str + ['$A'] + suffix_tokens_str, pair=
        prefix_tokens_str + ['$A', '$B'] + suffix_tokens_str,
        special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self
        .prefix_tokens + self.suffix_tokens)))
