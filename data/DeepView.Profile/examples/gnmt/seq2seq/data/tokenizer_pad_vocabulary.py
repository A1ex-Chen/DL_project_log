def pad_vocabulary(self, vocab, pad):
    """
        Pads vocabulary to a multiple of 'pad' tokens.

        :param vocab: list with vocabulary
        :param pad: integer
        """
    vocab_size = len(vocab)
    padded_vocab_size = (vocab_size + pad - 1) // pad * pad
    for i in range(0, padded_vocab_size - vocab_size):
        token = f'madeupword{i:04d}'
        vocab.append(token)
    assert len(vocab) % pad == 0
