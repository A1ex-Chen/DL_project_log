def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    return self.gpt2_tokenizer.decode(tokens)
