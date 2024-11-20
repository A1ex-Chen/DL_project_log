def test_full_tokenizer(self):
    tokenizer = TransfoXLTokenizer(vocab_file=self.vocab_file, lower_case=True)
    tokens = tokenizer.tokenize(u'<unk> UNwanted , running')
    self.assertListEqual(tokens, ['<unk>', 'unwanted', ',', 'running'])
    self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [0, 4, 8, 7])
