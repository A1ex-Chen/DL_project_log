def test_add_tokens_tokenizer(self):
    tokenizer = self.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    all_size = len(tokenizer)
    self.assertNotEqual(vocab_size, 0)
    self.assertEqual(vocab_size, all_size)
    new_toks = ['aaaaa bbbbbb', 'cccccccccdddddddd']
    added_toks = tokenizer.add_tokens(new_toks)
    vocab_size_2 = tokenizer.vocab_size
    all_size_2 = len(tokenizer)
    self.assertNotEqual(vocab_size_2, 0)
    self.assertEqual(vocab_size, vocab_size_2)
    self.assertEqual(added_toks, len(new_toks))
    self.assertEqual(all_size_2, all_size + len(new_toks))
    tokens = tokenizer.encode('aaaaa bbbbbb low cccccccccdddddddd l',
        add_special_tokens=False)
    out_string = tokenizer.decode(tokens)
    self.assertGreaterEqual(len(tokens), 4)
    self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
    self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
    new_toks_2 = {'eos_token': '>>>>|||<||<<|<<', 'pad_token':
        '<<<<<|||>|>>>>|>'}
    added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
    vocab_size_3 = tokenizer.vocab_size
    all_size_3 = len(tokenizer)
    self.assertNotEqual(vocab_size_3, 0)
    self.assertEqual(vocab_size, vocab_size_3)
    self.assertEqual(added_toks_2, len(new_toks_2))
    self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))
    tokens = tokenizer.encode(
        '>>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l',
        add_special_tokens=False)
    out_string = tokenizer.decode(tokens)
    self.assertGreaterEqual(len(tokens), 6)
    self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
    self.assertGreater(tokens[0], tokens[1])
    self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
    self.assertGreater(tokens[-2], tokens[-3])
    self.assertEqual(tokens[0], tokenizer.eos_token_id)
    self.assertEqual(tokens[-2], tokenizer.pad_token_id)
