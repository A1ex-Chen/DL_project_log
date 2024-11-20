def convert_tokens_to_string(self, tokens):
    """
        Converts a sequence of tokens (string) in a single string. Additionally, the split numbers are converted back
        into it's original form.
        """
    out_string = self.moses_detokenizer.detokenize(tokens)
    return detokenize_numbers(out_string).strip()
