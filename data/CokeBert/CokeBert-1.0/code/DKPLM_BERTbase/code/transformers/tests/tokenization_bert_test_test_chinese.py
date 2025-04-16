def test_chinese(self):
    tokenizer = BasicTokenizer()
    self.assertListEqual(tokenizer.tokenize(u'ah博推zz'), [u'ah', u'博', u'推',
        u'zz'])
