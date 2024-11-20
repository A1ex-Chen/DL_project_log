def test_basic_tokenizer_no_lower(self):
    tokenizer = BasicTokenizer(do_lower_case=False)
    self.assertListEqual(tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '),
        ['HeLLo', '!', 'how', 'Are', 'yoU', '?'])
