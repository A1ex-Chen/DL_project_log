def test_wordpiece_tokenizer(self):
    vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed',
        'wa', 'un', 'runn', '##ing']
    vocab = {}
    for i, token in enumerate(vocab_tokens):
        vocab[token] = i
    tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')
    self.assertListEqual(tokenizer.tokenize(''), [])
    self.assertListEqual(tokenizer.tokenize('unwanted running'), ['un',
        '##want', '##ed', 'runn', '##ing'])
    self.assertListEqual(tokenizer.tokenize('unwantedX running'), ['[UNK]',
        'runn', '##ing'])
