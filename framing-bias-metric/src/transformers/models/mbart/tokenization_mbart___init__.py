def __init__(self, *args, tokenizer_file=None, **kwargs):
    super().__init__(*args, tokenizer_file=tokenizer_file, **kwargs)
    self.sp_model_size = len(self.sp_model)
    self.lang_code_to_id = {code: (self.sp_model_size + i + self.
        fairseq_offset) for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)}
    self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
    self.cur_lang_code = self.lang_code_to_id['en_XX']
    self.fairseq_tokens_to_ids['<mask>'] = len(self.sp_model) + len(self.
        lang_code_to_id) + self.fairseq_offset
    self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
    self.fairseq_ids_to_tokens = {v: k for k, v in self.
        fairseq_tokens_to_ids.items()}
    self._additional_special_tokens = list(self.lang_code_to_id.keys())
    self.set_src_lang_special_tokens(kwargs.get('src_lang', 'en_XX'))
