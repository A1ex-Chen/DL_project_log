def test_basic_tokenizer_lower(self):
    tokenizer = BasicTokenizer(do_lower_case=True)
    self.assertListEqual(tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '),
        ['hello', '!', 'how', 'are', 'you', '?'])
    self.assertListEqual(tokenizer.tokenize(u'Héllo'), ['hello'])
