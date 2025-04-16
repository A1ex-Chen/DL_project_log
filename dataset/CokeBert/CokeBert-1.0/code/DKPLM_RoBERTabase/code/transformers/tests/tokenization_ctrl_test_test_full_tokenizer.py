def test_full_tokenizer(self):
    tokenizer = CTRLTokenizer(self.vocab_file, self.merges_file, **self.
        special_tokens_map)
    text = 'adapt react readapt apt'
    bpe_tokens = 'adapt re@@ a@@ c@@ t re@@ adapt apt'.split()
    tokens = tokenizer.tokenize(text)
    self.assertListEqual(tokens, bpe_tokens)
    input_tokens = tokens + [tokenizer.unk_token]
    input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
    self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens),
        input_bpe_tokens)
