def test_full_tokenizer_lower(self):
    tokenizer = TransfoXLTokenizer(lower_case=True)
    self.assertListEqual(tokenizer.tokenize(
        u' \tHeLLo ! how  \n Are yoU ?  '), ['hello', '!', 'how', 'are',
        'you', '?'])
