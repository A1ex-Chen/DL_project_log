def test_full_tokenizer(self):
    tokenizer = GPT2Tokenizer(self.vocab_file, self.merges_file, **self.
        special_tokens_map)
    text = 'lower newer'
    bpe_tokens = ['Ġlow', 'er', 'Ġ', 'n', 'e', 'w', 'er']
    tokens = tokenizer.tokenize(text, add_prefix_space=True)
    self.assertListEqual(tokens, bpe_tokens)
    input_tokens = tokens + [tokenizer.unk_token]
    input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
    self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens),
        input_bpe_tokens)
