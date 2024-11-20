def setUp(self):
    super(XLMTokenizationTest, self).setUp()
    vocab = ['l', 'o', 'w', 'e', 'r', 's', 't', 'i', 'd', 'n', 'w</w>',
        'r</w>', 't</w>', 'lo', 'low', 'er</w>', 'low</w>', 'lowest</w>',
        'newer</w>', 'wider</w>', '<unk>']
    vocab_tokens = dict(zip(vocab, range(len(vocab))))
    merges = ['l o 123', 'lo w 1456', 'e r</w> 1789', '']
    self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'vocab_file'])
    self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'merges_file'])
    with open(self.vocab_file, 'w') as fp:
        fp.write(json.dumps(vocab_tokens))
    with open(self.merges_file, 'w') as fp:
        fp.write('\n'.join(merges))
