def test_full_tokenizer(self):
    tokenizer = self.tokenizer_class(self.vocab_file)
    tokens = tokenizer.tokenize(u'UNwantéd,running')
    self.assertListEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing']
        )
    self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5,
        10, 8, 9])
