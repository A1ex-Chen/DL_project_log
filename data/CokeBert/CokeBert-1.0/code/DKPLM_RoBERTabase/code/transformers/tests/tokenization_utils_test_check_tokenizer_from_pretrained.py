@pytest.mark.slow
def check_tokenizer_from_pretrained(self, tokenizer_class):
    s3_models = list(tokenizer_class.max_model_input_sizes.keys())
    for model_name in s3_models[:1]:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        self.assertIsNotNone(tokenizer)
        self.assertIsInstance(tokenizer, tokenizer_class)
        self.assertIsInstance(tokenizer, PreTrainedTokenizer)
        for special_tok in tokenizer.all_special_tokens:
            if six.PY2:
                self.assertIsInstance(special_tok, unicode)
            else:
                self.assertIsInstance(special_tok, str)
            special_tok_id = tokenizer.convert_tokens_to_ids(special_tok)
            self.assertIsInstance(special_tok_id, int)
