def test_full_tokenizer(self):
    tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
    tokens = tokenizer.tokenize(u'This is a test')
    self.assertListEqual(tokens, [u'▁This', u'▁is', u'▁a', u'▁t', u'est'])
    self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [285, 46,
        10, 170, 382])
    tokens = tokenizer.tokenize(u'I was born in 92000, and this is falsé.')
    self.assertListEqual(tokens, [SPIECE_UNDERLINE + u'I', SPIECE_UNDERLINE +
        u'was', SPIECE_UNDERLINE + u'b', u'or', u'n', SPIECE_UNDERLINE +
        u'in', SPIECE_UNDERLINE + u'', u'9', u'2', u'0', u'0', u'0', u',', 
        SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE + u'this', 
        SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al', u's',
        u'é', u'.'])
    ids = tokenizer.convert_tokens_to_ids(tokens)
    self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 
        347, 3, 12, 66, 46, 72, 80, 6, 0, 4])
    back_tokens = tokenizer.convert_ids_to_tokens(ids)
    self.assertListEqual(back_tokens, [SPIECE_UNDERLINE + u'I', 
        SPIECE_UNDERLINE + u'was', SPIECE_UNDERLINE + u'b', u'or', u'n', 
        SPIECE_UNDERLINE + u'in', SPIECE_UNDERLINE + u'', u'<unk>', u'2',
        u'0', u'0', u'0', u',', SPIECE_UNDERLINE + u'and', SPIECE_UNDERLINE +
        u'this', SPIECE_UNDERLINE + u'is', SPIECE_UNDERLINE + u'f', u'al',
        u's', u'<unk>', u'.'])
