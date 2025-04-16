def __init__(self, vocab, unk_token, normalize_text=True):
    """
        Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
    self.vocab = vocab
    self.unk_token = unk_token
    self.normalize_text = normalize_text
