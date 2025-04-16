def setUp(self):
    super(XLNetTokenizationTest, self).setUp()
    tokenizer = XLNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
    tokenizer.save_pretrained(self.tmpdirname)
