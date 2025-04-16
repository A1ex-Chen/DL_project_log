def setUp(self):
    super(CTRLTokenizationTest, self).setUp()
    vocab = ['adapt', 're@@', 'a@@', 'apt', 'c@@', 't', '<unk>']
    vocab_tokens = dict(zip(vocab, range(len(vocab))))
    merges = ['#version: 0.2', 'a p', 'ap t</w>', 'r e', 'a d',
        'ad apt</w>', '']
    self.special_tokens_map = {'unk_token': '<unk>'}
    self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'vocab_file'])
    self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'merges_file'])
    with open(self.vocab_file, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(vocab_tokens) + '\n')
    with open(self.merges_file, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(merges))
