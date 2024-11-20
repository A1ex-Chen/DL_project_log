def setUp(self):
    super(GPT2TokenizationTest, self).setUp()
    vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'Ġ', 'Ġl',
        'Ġn', 'Ġlo', 'Ġlow', 'er', 'Ġlowest', 'Ġnewer', 'Ġwider', '<unk>']
    vocab_tokens = dict(zip(vocab, range(len(vocab))))
    merges = ['#version: 0.2', 'Ġ l', 'Ġl o', 'Ġlo w', 'e r', '']
    self.special_tokens_map = {'unk_token': '<unk>'}
    self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'vocab_file'])
    self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'merges_file'])
    with open(self.vocab_file, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(vocab_tokens) + '\n')
    with open(self.merges_file, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(merges))
