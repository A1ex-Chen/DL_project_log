def test_full_tokenizer(self):
    tokenizer = OpenAIGPTTokenizer(self.vocab_file, self.merges_file)
    text = 'lower'
    bpe_tokens = ['low', 'er</w>']
    tokens = tokenizer.tokenize(text)
    self.assertListEqual(tokens, bpe_tokens)
    input_tokens = tokens + ['<unk>']
    input_bpe_tokens = [14, 15, 20]
    self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens),
        input_bpe_tokens)
