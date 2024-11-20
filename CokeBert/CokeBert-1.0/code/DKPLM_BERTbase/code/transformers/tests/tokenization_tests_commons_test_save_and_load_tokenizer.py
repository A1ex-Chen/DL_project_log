def test_save_and_load_tokenizer(self):
    tokenizer = self.get_tokenizer()
    self.assertNotEqual(tokenizer.max_len, 42)
    tokenizer = self.get_tokenizer(max_len=42)
    before_tokens = tokenizer.encode(u'He is very happy, UNwantéd,running',
        add_special_tokens=False)
    with TemporaryDirectory() as tmpdirname:
        tokenizer.save_pretrained(tmpdirname)
        tokenizer = self.tokenizer_class.from_pretrained(tmpdirname)
        after_tokens = tokenizer.encode(u'He is very happy, UNwantéd,running',
            add_special_tokens=False)
        self.assertListEqual(before_tokens, after_tokens)
        self.assertEqual(tokenizer.max_len, 42)
        tokenizer = self.tokenizer_class.from_pretrained(tmpdirname, max_len=43
            )
        self.assertEqual(tokenizer.max_len, 43)
