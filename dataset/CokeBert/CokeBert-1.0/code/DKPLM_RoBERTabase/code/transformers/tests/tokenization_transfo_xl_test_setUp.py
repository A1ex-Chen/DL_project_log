def setUp(self):
    super(TransfoXLTokenizationTest, self).setUp()
    vocab_tokens = ['<unk>', '[CLS]', '[SEP]', 'want', 'unwanted', 'wa',
        'un', 'running', ',', 'low', 'l']
    self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES[
        'vocab_file'])
    with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
        vocab_writer.write(''.join([(x + '\n') for x in vocab_tokens]))
