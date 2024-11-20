def decode(self, tokens):
    """
        Decode list of tokens to text strings

        Args:
          tokens (:obj:`list<str>`): list of tokens.

        Returns:
          Text string corresponds to the input tokens.

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
          >>> tokenizer.decode(tokens)
          'Hello world!'
        """
    return self.bpe.decode([int(t) for t in tokens if t not in self.
        special_tokens])
