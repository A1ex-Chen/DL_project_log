def _parse_and_tokenize(self, inputs, **kwargs):
    """
        Parse arguments and tokenize, adding an EOS token at the end of the user input
        """
    inputs = self.tokenizer(inputs, add_special_tokens=False, padding=False
        ).get('input_ids', [])
    for input in inputs:
        input.append(self.tokenizer.eos_token_id)
    return inputs
