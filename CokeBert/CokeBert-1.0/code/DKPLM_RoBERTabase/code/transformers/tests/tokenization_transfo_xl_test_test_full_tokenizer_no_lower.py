def test_full_tokenizer_no_lower(self):
    tokenizer = TransfoXLTokenizer(lower_case=False)
    self.assertListEqual(tokenizer.tokenize(
        u' \tHeLLo ! how  \n Are yoU ?  '), ['HeLLo', '!', 'how', 'Are',
        'yoU', '?'])
