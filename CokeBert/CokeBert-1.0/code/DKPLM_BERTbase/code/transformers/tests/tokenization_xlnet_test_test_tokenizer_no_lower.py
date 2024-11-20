def test_tokenizer_no_lower(self):
    tokenizer = XLNetTokenizer(SAMPLE_VOCAB, do_lower_case=False)
    tokens = tokenizer.tokenize(u'I was born in 92000, and this is falsé.')
    self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE +
        u'was', SPIECE_UNDERLINE + u'b', u'or', u'n', SPIECE_UNDERLINE +
        u'in', SPIECE_UNDERLINE + u'', u'9', u'2', u'0', u'0', u'0', u',', 
        SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this', 
        SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u'se', u'.'])
