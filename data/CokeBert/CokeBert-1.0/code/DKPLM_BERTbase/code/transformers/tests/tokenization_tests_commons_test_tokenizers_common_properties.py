def test_tokenizers_common_properties(self):
    tokenizer = self.get_tokenizer()
    attributes_list = ['bos_token', 'eos_token', 'unk_token', 'sep_token',
        'pad_token', 'cls_token', 'mask_token']
    for attr in attributes_list:
        self.assertTrue(hasattr(tokenizer, attr))
        self.assertTrue(hasattr(tokenizer, attr + '_id'))
    self.assertTrue(hasattr(tokenizer, 'additional_special_tokens'))
    self.assertTrue(hasattr(tokenizer, 'additional_special_tokens_ids'))
    attributes_list = ['max_len', 'init_inputs', 'init_kwargs',
        'added_tokens_encoder', 'added_tokens_decoder']
    for attr in attributes_list:
        self.assertTrue(hasattr(tokenizer, attr))
