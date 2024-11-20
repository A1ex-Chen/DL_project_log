def tokenize(self, text):
    """
        Tokenizes a piece of text into characters.

        For example, :obj:`input = "apple""` wil return as output :obj:`["a", "p", "p", "l", "e"]`.

        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.

        Returns:
            A list of characters.
        """
    if self.normalize_text:
        text = unicodedata.normalize('NFKC', text)
    output_tokens = []
    for char in text:
        if char not in self.vocab:
            output_tokens.append(self.unk_token)
            continue
        output_tokens.append(char)
    return output_tokens
