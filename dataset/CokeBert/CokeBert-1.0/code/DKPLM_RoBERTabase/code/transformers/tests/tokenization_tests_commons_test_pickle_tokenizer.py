def test_pickle_tokenizer(self):
    tokenizer = self.get_tokenizer()
    self.assertIsNotNone(tokenizer)
    text = u'Munich and Berlin are nice cities'
    subwords = tokenizer.tokenize(text)
    with TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, u'tokenizer.bin')
        pickle.dump(tokenizer, open(filename, 'wb'))
        tokenizer_new = pickle.load(open(filename, 'rb'))
    subwords_loaded = tokenizer_new.tokenize(text)
    self.assertListEqual(subwords, subwords_loaded)
