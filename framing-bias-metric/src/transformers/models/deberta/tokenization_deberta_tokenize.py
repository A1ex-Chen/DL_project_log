def tokenize(self, text):
    """
        Convert an input text to tokens.

        Args:
          text (:obj:`str`): input text to be tokenized.

        Returns:
          A list of byte tokens where each token represent the byte id in GPT2 byte dictionary

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
        """
    bpe = self._encode(text)
    return [t for t in bpe.split(' ') if t]
