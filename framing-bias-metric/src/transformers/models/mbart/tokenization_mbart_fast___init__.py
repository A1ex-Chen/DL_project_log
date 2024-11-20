def __init__(self, *args, tokenizer_file=None, **kwargs):
    super().__init__(*args, tokenizer_file=tokenizer_file, **kwargs)
    self.cur_lang_code = self.convert_tokens_to_ids('en_XX')
    self.set_src_lang_special_tokens(kwargs.get('src_lang', 'en_XX'))
    self.add_special_tokens({'additional_special_tokens':
        FAIRSEQ_LANGUAGE_CODES})
